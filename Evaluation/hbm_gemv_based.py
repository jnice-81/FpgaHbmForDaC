from typing import List
import dace
from dace import subsets
from dace import memlet
from dace import dtypes
from dace.sdfg.sdfg import InterstateEdge, SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import optimizer
from dace.transformation.interstate.sdfg_nesting import NestSDFG
from dace.transformation.optimizer import Optimizer
from dace.transformation.interstate import InlineSDFG, FPGATransformSDFG
from dace.transformation.dataflow import StreamingMemory
from dace.sdfg import graph, nodes, propagation, utils
from dace.libraries.blas.nodes import dot

from helper import *
from hbm_transform import HbmTransform
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
    lib_node.expand(sdfg, state, tile_size_x=32, tile_size_y=1024)
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
            strform.apply(sdfg)

    return sdfg

#hbm_gemv_sdfg(32).view()
    
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