from typing import List
import dace
from dace import subsets
from dace import memlet
from dace import dtypes
from dace.sdfg.sdfg import InterstateEdge, SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import optimizer
from dace.transformation.interstate.fpga_transform_state import FPGATransformState
from dace.transformation.interstate.sdfg_nesting import NestSDFG
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import InlineSDFG, FPGATransformSDFG
from dace.transformation.dataflow import StreamingMemory, MapInterchange, StripMining
from dace.sdfg import graph, nodes, propagation, utils
from dace.libraries.blas.nodes import dot

from helper import *
from hbm_transform import HbmTransform, hbm_module_distribute, unroll_map
from hbm_bank_split import HbmBankSplit
from hbm_transform import set_shape
from hbm_transform import transform_sdfg_for_hbm
from hbm_transform import all_innermost_edges

def simple_gemv_sdfg(M, N, tile_size_x, tile_size_y):
    @dace.program
    def gemv(A: dace.float32[M, N], x: dace.float32[N], y: dace.float32[M]):
        y[:] = A @ x
    sdfg = gemv.to_sdfg()
    state = sdfg.states()[0]
    lib_node = get_first_node(state, lambda x: isinstance(x, nodes.LibraryNode))
    lib_node.expand(sdfg, state)
    lib_node = get_first_node(state, lambda x: isinstance(x, nodes.LibraryNode))
    lib_node.implementation = "FPGA_TilesByColumn"
    lib_node.expand(sdfg, state, tile_size_x=tile_size_x, tile_size_y=tile_size_y)
    sdfg.apply_strict_transformations()

    # Enable buffering of A. Uses automatic port widening, since this is easier in this case
    # because of the later need of the single elements
    vec_size = 16
    A_acc = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "A")
    A_edge_out = state.out_edges(A_acc)[0]
    path_nodes = get_nodes_of_path(state.memlet_path(A_edge_out))
    state.remove_memlet_path(A_edge_out, False)
    buffer_map_y_entry, buffer_map_y_exit = state.add_map("buffer_y", {"buf_y": f"0:{tile_size_y}"})
    buffer_map_x_entry, buffer_map_x_exit = state.add_map("buffer_x", {"buf_x": f"0:{tile_size_x//vec_size}"})
    buffer_map_unroll_entry, buffer_map_unroll_exit = state.add_map("buffer_unroll", {"buf_ix": f"0:{vec_size}"})
    buffer_map_unroll_entry.map.unroll = True
    read_a_tasklet = state.add_tasklet("read_A", set(["_in"]), set(["_out"]), "_out = _in")
    sdfg.add_array("A_buffer", [tile_size_y, tile_size_x], dace.float32, dtypes.StorageType.FPGA_Local, True)
    A_buffer_acc = state.add_access("A_buffer")

    state.add_memlet_path(*path_nodes[0:3], buffer_map_y_entry, buffer_map_x_entry, buffer_map_unroll_entry,
        read_a_tasklet, memlet=memlet.Memlet(f"A[{tile_size_y}*ty+buf_y, {tile_size_x}*tx + buf_x*{vec_size}+buf_ix]"), dst_conn="_in")
    state.add_memlet_path(read_a_tasklet, buffer_map_unroll_exit, buffer_map_x_exit, buffer_map_y_exit, A_buffer_acc, 
        memlet=memlet.Memlet(f"A_buffer[buf_y, buf_x*{vec_size} + buf_ix]"), src_conn="_out")
    gemvT: nodes.Tasklet = path_nodes[-1]
    gemvT.add_in_connector("A_in", dace.float32)
    state.add_memlet_path(A_buffer_acc,  *path_nodes[3:], 
        memlet=memlet.Memlet("A_buffer[iy, ix]"), dst_conn="A_in", )

    # unroll the compute around gemv
    cmp_gemv = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "y")
    cmp_gemv = StripMining.apply_to(sdfg, {"skew": True, "divides_evenly": True, 
        "tiling_type": dtypes.TilingType.CeilRange, "tile_size": vec_size}, _map_entry=cmp_gemv)
    cmp_gemv.range = subsets.Range.from_string(f"0:{tile_size_y}//{vec_size}") # sympy does not get ceil
    cmp_gemv = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "y")
    cmp_gemv.map.schedule = dtypes.ScheduleType.FPGA_Device
    cmp_gemv.map.unroll = True

    return sdfg


def hbm_gemv_sdfg(banks_A: int):
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = simple_gemv_sdfg(M, N, 1024, 32)
    state = sdfg.states()[0]
    
    map_node = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "y_tiles")
    desc_A = sdfg.arrays["A"]
    desc_x = sdfg.arrays["x"]
    desc_y = sdfg.arrays["y"]
    desc_A.location["memorytype"] = "HBM"
    desc_A.location["bank"] = f"0:{banks_A}"
    desc_x.location["memorytype"] = "DDR"
    desc_x.location["bank"] = f"0"
    desc_y.location["memorytype"] = "DDR"
    desc_y.location["bank"] = f"1"
    HbmTransform.apply_to(sdfg, _map_entry=map_node)
    
    for strform in optimizer.Optimizer(sdfg).get_pattern_matches(patterns=StreamingMemory):
        where = state.nodes()[strform.subgraph[strform.access]].data
        if where == "x" or where == "y":
            strform.buffer_size = 32
            strform.apply(sdfg)

    y_write_entry = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__swrite_y_0" and
        x.params[0] == "k")
    x_read_entry = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__sread_x_0" and
        x.params[0] == "k")
    hbm_module_distribute(sdfg, state, y_write_entry, "y_0", banks_A, False, 4)
    hbm_module_distribute(sdfg, state, x_read_entry, "x_0", banks_A, True, 4)

    return sdfg

def only_hbm_gemv_sdfg(banks_A: int):
    sdfg = hbm_gemv_sdfg(banks_A)
    
    # Applying fpga transform here does not work because streams are not in a map. We still want to do it so we force.
    sdfg.apply_transformations(NestSDFG)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.Default
    fpga_xform = FPGATransformSDFG(sdfg.sdfg_id, -1, {}, -1)
    fpga_xform.apply(sdfg)
    sdfg.apply_transformations_repeated(InlineSDFG)
    
    distribute_along_dim0(sdfg, ["A"])

    return sdfg
