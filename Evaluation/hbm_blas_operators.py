from typing import List
import dace
from dace import subsets
from dace import memlet
from dace import dtypes
from dace.sdfg.sdfg import InterstateEdge, SDFG
from dace.sdfg.state import SDFGState
from dace.transformation.interstate.sdfg_nesting import NestSDFG
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import InlineSDFG, FPGATransformSDFG
from dace.transformation.dataflow import StripMining
from dace.sdfg import graph, nodes, propagation, utils
from dace.libraries.blas.nodes import dot

from hbm_transform import HbmTransform
from hbm_bank_split import HbmBankSplit
from hbm_transform import set_shape
from hbm_transform import transform_sdfg_for_hbm
from hbm_transform import all_innermost_edges

######## Helpers

def get_first_node(state: SDFGState, cond):
    for node, state in state.all_nodes_recursive():
        if cond(node):
            return node

def distribute_along_dim0(sdfg, array_list: List[str]):
    for array in array_list:
        desc = sdfg.arrays[array]
        if len(desc.shape) > 2:
            new_shape = [desc.shape[0] * desc.shape[1], *desc.shape[2:]]
        else:
            new_shape = [desc.shape[0] * desc.shape[1]]
        set_shape(desc, new_shape)
    for match in Optimizer(sdfg).get_pattern_matches(patterns=HbmBankSplit):
        match.apply(sdfg)

######## Simple base versions of the pure blas applications without HBM use

def simple_vadd_sdfg(N):
    @dace.program
    def axpy(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                xin << x[i]
                yin << y[i]
                zout >> z[i]
                zout = xin + yin
    sdfg = axpy.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(StripMining, {"tile_size": 1200, "divides_evenly": True})
    map = get_first_node(sdfg.start_state, lambda x: isinstance(x, nodes.MapEntry) and x.map.params[0] == "i")
    map.map.schedule = dtypes.ScheduleType.FPGA_Device
    StripMining.apply_to(sdfg, {"tile_size": 12, "divides_evenly": True}, _map_entry=map)
    map = get_first_node(sdfg.start_state, lambda x: isinstance(x, nodes.MapEntry) and x.map.params[0] == "i")
    map.map.schedule = dtypes.ScheduleType.FPGA_Device
    map.map.unroll = True
    
    return sdfg

def simple_gemv_sdfg(M, N):
    @dace.program
    def gemv(A: dace.float32[M, N], x: dace.float32[N], y: dace.float32[M]):
        y[:] = A @ x
    sdfg = gemv.to_sdfg()
    state = sdfg.states()[0]
    lib_node = get_first_node(state, lambda x: isinstance(x, nodes.LibraryNode))
    lib_node.expand(sdfg, state)
    lib_node = get_first_node(state, lambda x: isinstance(x, nodes.LibraryNode))
    lib_node.implementation = "FPGA_TilesByColumn"
    lib_node.expand(sdfg, state, tile_size_x=32, tile_size_y=1024)
    sdfg.apply_strict_transformations()
    return sdfg

def simple_dot_sdfg(N): #repair broken other subset
    sdfg: SDFG = SDFG("dot")
    state = sdfg.add_state()

    sdfg.add_array("x", [N], dace.float32, dtypes.StorageType.FPGA_Global)
    sdfg.add_array("y", [N], dace.float32, dtypes.StorageType.FPGA_Global)
    sdfg.add_array("result", [1], dace.float32, dtypes.StorageType.FPGA_Global)

    lib_node = dot.Dot("dot")
    state.add_node(lib_node)
    read_x = state.add_read("x")
    read_y = state.add_read("y")
    write_result = state.add_write("result")
    state.add_edge(read_x, None, lib_node, "_x", memlet.Memlet("x"))
    state.add_edge(read_y, None, lib_node, "_y", memlet.Memlet("y"))
    state.add_edge(lib_node, "_result", write_result, None, memlet.Memlet(f"result"))
    lib_node.implementation = "FPGA_PartialSums"
    lib_node.expand(sdfg, state, partial_width=16)
    sdfg.arrays["x"].storage = dtypes.StorageType.Default
    sdfg.arrays["y"].storage = dtypes.StorageType.Default
    sdfg.arrays["result"].storage = dtypes.StorageType.Default

    strip_map = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "stream")
    for nsdfg in sdfg.all_sdfgs_recursive():
        if nsdfg.states()[0].label == "stream":
            StripMining.apply_to(nsdfg, {"tile_size": 1200, "divides_evenly": True}, _map_entry=strip_map)
            state = nsdfg.start_state
            tile_map = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "stream"
                and x.map.params[0] == "i")
            tile_map.map.schedule = dtypes.ScheduleType.FPGA_Device
            StripMining.apply_to(nsdfg, {"tile_size": 12, "divides_evenly": True}, _map_entry=tile_map)
            inner_tile_map = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "stream"
                and x.map.params[0] == "i")
            inner_tile_map.map.schedule = dtypes.ScheduleType.FPGA_Device
            inner_tile_map.map.unroll = True

            break
    
    return sdfg

######### On Device HBM-implementations of pure blas

def hbm_axpy_sdfg(banks_per_input: int):
    N = dace.symbol("N")
    sdfg = simple_vadd_sdfg(N)

    map = get_first_node(sdfg.start_state, lambda x: isinstance(x, nodes.MapEntry) and x.map.params[0] == "tile_i")
    banks = {"x": ("HBM", f"0:{banks_per_input}", [banks_per_input]), 
        "y": ("HBM", f"{banks_per_input}:{2*banks_per_input}", [banks_per_input]),
        "z": ("HBM", f"{2*banks_per_input}:{3*banks_per_input}", [banks_per_input])}
    transform_sdfg_for_hbm(sdfg, ("k", banks_per_input), banks, {(map, 0): banks_per_input})

    return sdfg

def hbm_gemv_sdfg(banks_A: int, no_split_y = True):
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = simple_gemv_sdfg(M, N)
    state = sdfg.states()[0]
    
    if no_split_y:
        map_node = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "y_tiles")
        y_access = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "y")
        edge_to_modify = next(all_innermost_edges(state, y_access))
        edge_to_modify.data.subset = subsets.Range.from_string(f"k*(M/{banks_A}) + 1024*ty + iy")
        transform_sdfg_for_hbm(sdfg, ("k", banks_A), {"A": ("HBM", f"0:{banks_A}", [banks_A, 1]),
            "x": ("HBM", "30", None), "y": ("HBM", "31", None)}, {(map_node.map, 0): banks_A})
        propagation.propagate_memlets_sdfg(sdfg)
    else:
        sdfg.arrays["y"].location["memorytype"] = "HBM"
        sdfg.arrays["y"].location["bank"] = f"0:{banks_A}"
        sdfg.apply_transformations(HbmTransform)

    return sdfg

def hbm_dot_sdfg(banks_per_input: int):
    N = dace.symbol("N")

    sdfg = simple_dot_sdfg(N)
    state = sdfg.states()[0]

    for edge, state in sdfg.all_edges_recursive():
        if isinstance(edge, graph.MultiConnectorEdge):
            if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == "_result":
                edge.data.other_subset = subsets.Range.from_string("k")
                set_shape(state.parent.arrays["_result"], [banks_per_input])
            if isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == "result":
                #one cannot update the other_subset. Leads to problems with out of bounds checking
                #edge.data.other_subset = subsets.Range.from_string("k")
                set_shape(state.parent.arrays["result"], [banks_per_input])

    array_banks = {"x": ("HBM", f"0:{banks_per_input}", [banks_per_input]), 
                "y": ("HBM", f"{banks_per_input}:{2*banks_per_input}", [banks_per_input]),
                "result": ("DDR", "0", None)}
    div_map = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "stream"
        and x.map.params[0] == "tile_i")
    transform_sdfg_for_hbm(sdfg, ("k", banks_per_input), 
        array_banks, {(div_map.map, 0): banks_per_input}, True)

    return sdfg

######### Full implementations of pure blas applications

def only_hbm_axpy_sdfg(banks_per_input: int):
    sdfg = hbm_axpy_sdfg(banks_per_input)
    sdfg.apply_fpga_transformations()
    sdfg.apply_transformations_repeated(InlineSDFG)
    z_access1 = get_first_node(sdfg.start_state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "z")
    sdfg.start_state.remove_nodes_from([sdfg.start_state.out_edges(z_access1)[0].dst, z_access1])
    distribute_along_dim0(sdfg, ["x", "y", "z"])

    return sdfg

def only_hbm_gemv_sdfg(banks_A: int, no_split_y = True):
    sdfg = hbm_gemv_sdfg(banks_A, no_split_y)
    sdfg.apply_fpga_transformations()
    sdfg.apply_transformations_repeated(InlineSDFG)
    if not no_split_y:
        distribute_along_dim0(sdfg, ["y"])
    src_node = get_first_node(sdfg.start_state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "A")
    dst_node = get_first_node(sdfg.start_state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "fpga_A")
    desc_A = sdfg.arrays["A"]
    set_shape(desc_A, [desc_A.shape[1] * banks_A, desc_A.shape[2]])
    HbmBankSplit.apply_to(sdfg, {"split_array_info": [banks_A, 1]}, _src_node=src_node, _dst_node=dst_node)

    return sdfg

def _modify_dot_host_side(sdfg, start_state, end_state):
    # Add final reduction
    state = end_state
    host_result = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "result")
    sum_up = state.add_reduce("lambda a, b : a + b", None, 0)
    sdfg.add_array("final_result", [1], dace.float32)
    host_final = state.add_access("final_result")
    state.add_edge(host_result, None, sum_up, None, memlet.Memlet("result"))
    state.add_edge(sum_up, None, host_final, None, memlet.Memlet("final_result[0]"))
    sum_up.expand(sdfg, state)
    sdfg.apply_transformations(InlineSDFG)

    # Remove copy result
    state = start_state
    access_result_start = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "result")
    state.remove_nodes_from([state.out_edges(access_result_start)[0].dst, access_result_start])

    sdfg.arrays["result"].transient = True

def only_hbm_dot_sdfg(banks_per_input: int):
    sdfg = hbm_dot_sdfg(banks_per_input)
    sdfg.apply_fpga_transformations()
    sdfg.apply_transformations_repeated(InlineSDFG)
    distribute_along_dim0(sdfg, ["x", "y"])

    _modify_dot_host_side(sdfg, sdfg.start_state, sdfg.states()[2])

    return sdfg

def hbm_axpy_dot(banks_per_input: int):
    N = dace.symbol("N")
    axpy_sdfg = simple_vadd_sdfg(N)
    dot_sdfg = simple_dot_sdfg(N)

    sdfg = SDFG("axpydot")
    state = sdfg.add_state()

    sdfg.add_array("axpy_x", [N], dace.float32)
    sdfg.add_array("axpy_y", [N], dace.float32)
    sdfg.add_array("dot_y", [N], dace.float32)
    sdfg.add_array("middle", [N], dace.float32, transient=True)
    sdfg.add_array("result", [10], dace.float32)
    
    acc_axpy_x = state.add_access("axpy_x")
    acc_axpy_y = state.add_access("axpy_y")
    acc_dot_y = state.add_access("dot_y")
    acc_middle = state.add_access("middle")
    acc_result = state.add_access("result")

    axpynode = state.add_nested_sdfg(axpy_sdfg, sdfg, set(["x", "y", "z"]), set(["z"]), {"N": N})
    dotnode = state.add_nested_sdfg(dot_sdfg, sdfg, set(["x", "y", "result"]), set(["result"]), {"N": N})

    acc_middle_dummy = state.add_access("middle")
    acc_middle_dummy_2 = state.add_access("middle")
    acc_result_dummy = state.add_access("result")

    state.add_edge(acc_axpy_x, None, axpynode, "x", memlet.Memlet("axpy_x"))
    state.add_edge(acc_axpy_y, None, axpynode, "y", memlet.Memlet("axpy_y"))
    state.add_edge(acc_middle_dummy, None, axpynode, "z", memlet.Memlet("middle"))
    state.add_edge(axpynode, "z", acc_middle, None, memlet.Memlet("middle"))

    state.add_edge(acc_middle_dummy_2, None, dotnode, "x", memlet.Memlet("middle"))
    state.add_edge(acc_dot_y, None, dotnode, "y", memlet.Memlet("dot_y"))
    state.add_edge(acc_result_dummy, None, dotnode, "result", memlet.Memlet("result"))
    state.add_edge(dotnode, "result", acc_result, None, memlet.Memlet("result"))

    #sdfg.apply_fpga_transformations(validate=False)
    sdfg.apply_transformations_repeated(InlineSDFG)

    def _nodes_from_path(path):
        nodes = [path[0].src]
        for edge in path:
            nodes.append(edge.dst)
        return nodes

    sdfg.add_stream("connect", dtypes.float32, 
        storage=dtypes.StorageType.FPGA_Local, transient=True)
    acc_connect = state.add_access("connect")
    old_acc_node = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "middle")
    old_edge = state.in_edges(old_acc_node)[0]
    old_path = state.memlet_path(old_edge)
    old_path_nodes = _nodes_from_path(old_path)
    old_path_nodes.pop()
    old_path_nodes.append(acc_connect)
    state.add_memlet_path(*old_path_nodes, memlet=memlet.Memlet("connect[0]"),
        src_conn="zout")
    state.remove_memlet_path(old_edge)

    old_acc_node = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "middle")
    old_edge = state.out_edges(old_acc_node)[0]
    old_path = state.memlet_path(old_edge)
    old_path_nodes = _nodes_from_path(old_path)
    old_path_nodes.pop(0)
    old_path_nodes.insert(0, acc_connect)
    state.add_memlet_path(*old_path_nodes, memlet=memlet.Memlet("connect[0]"),
        dst_conn="_in")
    state.remove_memlet_path(old_edge)

    # Fpga transform cannot be applied here, because stream is not in a map, and because there
    # are FPGA storagetypes and schedules around. However since the actual application of 
    # FPGATransform works non-destructive we just force application here
    fpga_xform = FPGATransformSDFG(sdfg.sdfg_id, -1, {}, -1)
    fpga_xform.apply(sdfg)
    sdfg.apply_transformations(InlineSDFG)

    return sdfg

#sdfg = hbm_axpy_dot(10)
#sdfg.view()
#sdfg.compile()