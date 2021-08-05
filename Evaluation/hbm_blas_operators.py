from typing import List
import dace
from dace import subsets
from dace import memlet
from dace import dtypes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation.interstate.sdfg_nesting import NestSDFG
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import InlineSDFG
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

def simple_axpy_sdfg(N, vec_size):
    @dace.program
    def axpy(x: dace.vector(dace.float32, vec_size)[N / vec_size], y: dace.vector(dace.float32, vec_size)[N / vec_size]):
        for i in dace.map[0:N/vec_size]:
            with dace.tasklet:
                xin << x[i]
                yin << y[i]
                yout >> y[i]
                yout = xin + yin
    sdfg = axpy.to_sdfg()
    sdfg.apply_strict_transformations()
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

def simple_dot_sdfg(N, vec_size):
    sdfg: SDFG = SDFG("dot")
    state = sdfg.add_state()

    sdfg.add_array("x", [N/vec_size], dace.vector(dace.float32, vec_size), dtypes.StorageType.FPGA_Global)
    sdfg.add_array("y", [N/vec_size], dace.vector(dace.float32, vec_size), dtypes.StorageType.FPGA_Global)
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
    lib_node.expand(sdfg, state)
    sdfg.arrays["x"].storage = dtypes.StorageType.Default
    sdfg.arrays["y"].storage = dtypes.StorageType.Default
    sdfg.arrays["result"].storage = dtypes.StorageType.Default

    return sdfg

######### On Device HBM-implementations of pure blas

def hbm_axpy_sdfg(vec_size: int, banks_per_input: int):
    N = dace.symbol("N")
    sdfg = simple_axpy_sdfg(N, vec_size)

    sdfg.arrays["x"].location["memorytype"] = "HBM"
    sdfg.arrays["x"].location["bank"] = f"0:{banks_per_input}"
    sdfg.apply_transformations(HbmTransform)
    """
    set_shape(sdfg.arrays["x"], [N/vec_size])
    set_shape(sdfg.arrays["y"], [N/vec_size])
    for match in Optimizer(sdfg).get_pattern_matches(patterns=HbmBankSplit):
        match.apply(sdfg)
    """
    
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

def hbm_dot_sdfg(vec_size: int, banks_per_input: int):
    N = dace.symbol("N")

    sdfg = simple_dot_sdfg(N, vec_size)
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
    div_map = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "stream")
    transform_sdfg_for_hbm(sdfg, ("k", banks_per_input), 
        array_banks, {(div_map.map, 0): banks_per_input}, True)

    return sdfg

######### Full implementations of pure blas applications

def only_hbm_axpy_sdfg(vec_size: int, banks_per_input: int):
    sdfg = hbm_axpy_sdfg(vec_size, banks_per_input)
    sdfg.apply_fpga_transformations()
    sdfg.apply_transformations_repeated(InlineSDFG)
    distribute_along_dim0(sdfg, ["x", "y"])

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

def only_hbm_dot_sdfg(vec_size: int, banks_per_input: int):
    sdfg = hbm_dot_sdfg(vec_size, banks_per_input)
    sdfg.apply_fpga_transformations()
    sdfg.apply_transformations_repeated(InlineSDFG)
    distribute_along_dim0(sdfg, ["x", "y"])

    state = sdfg.states()[2]
    host_result = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "result")
    sum_up = state.add_reduce("lambda a, b : a + b", None, None)
    sdfg.add_array("final_result", [1], dace.float32)
    host_final = state.add_access("final_result")
    state.add_edge(host_result, None, sum_up, None, memlet.Memlet("result"))
    state.add_edge(sum_up, None, host_final, None, memlet.Memlet("final_result[0]"))
    sum_up.expand(sdfg, state)
    sdfg.apply_transformations(InlineSDFG)

    return sdfg


#only_hbm_axpy_sdfg(1, 2)
#only_hbm_gemv_sdfg(2, True)
#only_hbm_dot_sdfg(1, 2)