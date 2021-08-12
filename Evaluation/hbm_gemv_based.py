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