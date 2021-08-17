import dace
from dace import dtypes
from dace.memlet import Memlet
from dace.libraries import blas
from dace.transformation.interstate import InlineSDFG, FPGATransformSDFG
from dace.transformation.dataflow import StreamingMemory, MapInterchange
from dace.transformation.interstate.sdfg_nesting import NestSDFG

from helper import *
from hbm_transform import HbmTransform, hbm_module_distribute, unroll_map
from hbm_bank_split import HbmBankSplit
from hbm_transform import set_shape
from hbm_transform import transform_sdfg_for_hbm
from hbm_transform import all_innermost_edges
from dace.transformation.optimizer import Optimizer

def pure_graph_ger_sdfg(implementation, dtype, veclen):
    #copied from the dace tests
    m = dace.symbol("m")
    n = dace.symbol("n")
    vtype = dace.vector(dtype, veclen)

    sdfg = dace.SDFG("ger_test")

    state = sdfg.add_state("ger")

    sdfg.add_symbol("alpha", dtype)

    sdfg.add_array("x", shape=[m], dtype=dtype)
    sdfg.add_array("y", shape=[n / veclen], dtype=vtype)
    sdfg.add_array("A", shape=[m, n / veclen], dtype=vtype)
    sdfg.add_array("res", shape=[m, n / veclen], dtype=vtype)

    x = state.add_read("x")
    y = state.add_read("y")
    A = state.add_read("A")
    res = state.add_write("res")

    ger_node = blas.Ger(name="ger")
    ger_node.implementation = implementation

    state.add_memlet_path(x, ger_node, dst_conn="_x", memlet=Memlet("x[0:m]"))
    state.add_memlet_path(y,
                          ger_node,
                          dst_conn="_y",
                          memlet=Memlet(f"y[0:n/{veclen}]"))
    state.add_memlet_path(A,
                          ger_node,
                          dst_conn="_A",
                          memlet=Memlet(f"A[0:m, 0:n/{veclen}]"))
    state.add_memlet_path(ger_node,
                          res,
                          src_conn="_res",
                          memlet=Memlet(f"res[0:m, 0:n/{veclen}]"))

    return ger_node, state, sdfg

def hbm_ger_sdfg(banks_A, tile_size_y, tile_size_x):
    lib_node, state, sdfg = pure_graph_ger_sdfg("FPGA", dace.float32, 8)
    lib_node.alpha = "alpha"
    lib_node.expand(sdfg, state, tile_size_x=tile_size_x, tile_size_y=tile_size_y)
    sdfg.apply_transformations(InlineSDFG)

    app_map = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label=="x_tiles")
    transform_sdfg_for_hbm(sdfg, ("k", banks_A), 
        {"A": ("HBM", f"0:{banks_A}", [banks_A, 1]), "x": ("DDR", "0", None), "y": ("DDR", "1", None),
             "res": ("HBM", f"{banks_A}:{2*banks_A}", [banks_A, 1])},
        {(app_map, 0): banks_A})
    sdfg.apply_transformations(InlineSDFG)
    state = sdfg.start_state
    x_acc = get_first_node(state, lambda x: isinstance(x, nodes.AccessNode) and x.data == "x")
    update_access(state, x_acc, "x", Memlet(f"x[ix + tx*{tile_size_x} + k*{app_map.map.range[0][1] + 1}]"))

    for name in ["A", "x", "y", "res"]:
        desc = sdfg.arrays[name]
        desc.storage = dtypes.StorageType.FPGA_Global
    for strform in Optimizer(sdfg).get_pattern_matches(patterns=StreamingMemory):
        where = state.nodes()[strform.subgraph[strform.access]].data
        if where == "x" or where == "y":
            strform.buffer_size = 32
            strform.apply(sdfg)
    
    state = sdfg.start_state
    x_str = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label == "__sread_x_0" and x.params[0] == "k")
    hbm_module_distribute(sdfg, state, x_str, "x_0", banks_A, True, 8)
    y_str = get_first_node(state, lambda x: isinstance(x, nodes.MapEntry) and x.label=="__sread_y_0" and x.params[0] == "k")
    hbm_module_distribute(sdfg, state, y_str, "y_0", banks_A, True, 8)

    sdfg.apply_transformations(NestSDFG)
    for desc in sdfg.arrays.values():
        desc.storage = dtypes.StorageType.Default

    fpga_xform = FPGATransformSDFG(sdfg.sdfg_id, -1, {}, -1)
    fpga_xform.apply(sdfg)
    sdfg.apply_transformations_repeated(InlineSDFG)

    distribute_along_dim0(sdfg, ("A", "res"))
    return sdfg