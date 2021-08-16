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
from dace.transformation.dataflow import StreamingMemory, MapInterchange
from dace.sdfg import graph, nodes, propagation, utils
from dace.libraries.blas.nodes import dot

from helper import *
from hbm_transform import HbmTransform, unroll_map
from hbm_bank_split import HbmBankSplit
from hbm_transform import set_shape
from hbm_transform import transform_sdfg_for_hbm
from hbm_transform import all_innermost_edges

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
    lib_node.expand(sdfg, state, tile_size_x=1024, tile_size_y=32)
    sdfg.apply_strict_transformations()
    return sdfg

def hbm_gemv_sdfg(banks_A: int):
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = simple_gemv_sdfg(M, N)
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

    # Rewrite streams such that they avoid global logic while keeping the number of interfaces small
    feed_count = 4
    y_write_entry = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__swrite_y_0" and
        x.params[0] == "k")
    unroll_map(sdfg, state, y_write_entry, feed_count, "bank")
    while True:
        y_write_entry = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__swrite_y_0" and
            x.params[0] == "k")
        next_node = state.out_edges(y_write_entry)[0].dst
        if not isinstance(next_node, nodes.MapEntry):
            break
        MapInterchange.apply_to(sdfg, outer_map_entry=y_write_entry, inner_map_entry=next_node)
    x_read_entry = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__sread_x_0" and
        x.params[0] == "k")
    unroll_map(sdfg, state, x_read_entry, feed_count, "bank")
    while True:
        x_read_entry = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__sread_x_0" and
            x.params[0] == "k")
        next_node = state.out_edges(x_read_entry)[0].dst
        if not isinstance(next_node, nodes.MapEntry):
            break
        MapInterchange.apply_to(sdfg, outer_map_entry=x_read_entry, inner_map_entry=next_node)

    sdfg.arrays["y_0"].shape = (banks_A, )
    y_stream_read = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "y_0" and 
        state.out_degree(x) > 0)
    y_stream_write = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "y_0" and 
        state.out_degree(x) == 0)
    state.memlet_path(state.out_edges(y_stream_read)[0])[-1].data.subset = subsets.Range.from_string(f"k+{banks_A//feed_count}*bank")
    state.memlet_path(state.in_edges(y_stream_write)[0])[0].data.subset = subsets.Range.from_string("k")

    sdfg.arrays["x_0"].shape = (banks_A, )
    x_stream_read = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "x_0" and 
        state.out_degree(x) > 0)
    x_stream_write = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "x_0" and 
        state.out_degree(x) == 0)
    state.memlet_path(state.out_edges(x_stream_read)[0])[-1].data.subset = subsets.Range.from_string("k")
    state.memlet_path(state.in_edges(x_stream_write)[0])[0].data.subset = subsets.Range.from_string(f"k+{banks_A//feed_count}*bank")

    propagation.propagate_memlets_sdfg(sdfg)

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