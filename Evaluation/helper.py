from typing import List

from dace import memlet
from dace.sdfg.state import SDFGState
from dace.transformation.optimizer import Optimizer

from hbm_bank_split import HbmBankSplit
from hbm_transform import set_shape
from dace.sdfg import graph, nodes, propagation, utils

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

def update_access(state: SDFGState, old_acc_node: nodes.AccessNode, new_data: str, new_memlet: memlet.Memlet):
    old_edge = state.all_edges(old_acc_node)[0]
    path = state.memlet_path(old_edge)
    if path[0] == old_edge:
        path[-1].data = new_memlet
    else:
        path[0].data = new_memlet
    old_acc_node.data = new_data

def get_nodes_of_path(path: List[graph.MultiConnectorEdge]):
    nodes = []
    nodes.append(path[0].src)
    for e in path:
        nodes.append(e.dst)
    return nodes