from core.methods.node.base import NodeClassification
from core.methods.node.gap.gap_inf import GAP
from core.methods.node.gap.gap_edp import EdgePrivGAP
from core.methods.node.gap.gap_ndp import NodePrivGAP
from core.methods.node.sage.sage_inf import SAGE
from core.methods.node.sage.sage_edp import EdgePrivSAGE
from core.methods.node.sage.sage_ndp import NodePrivSAGE
from core.methods.node.mlp.mlp import MLP
from core.methods.node.mlp.mlp_dp import PrivMLP
from core.methods.node.dpdgc.dpdgc_inf import DPDGC
from core.methods.node.dpdgc.dpdgc_edp import EdgePrivDPDGC
from core.methods.node.dpdgc.dpdgc_ndp import NodePrivDPDGC
from core.methods.node.dpdgc.dpdgc_kndp import KNeighborPrivDPDGC

supported_methods = {
    'gap-inf':  GAP,
    'gap-edp':  EdgePrivGAP,
    'gap-ndp':  NodePrivGAP,
    'sage-inf': SAGE,
    'sage-edp': EdgePrivSAGE,
    'sage-ndp': NodePrivSAGE,
    'mlp':      MLP,
    'mlp-dp':   PrivMLP,
    'dpdgc-inf': DPDGC,
    'dpdgc-edp': EdgePrivDPDGC,
    'dpdgc-ndp': NodePrivDPDGC,
    'dpdgc-kndp': KNeighborPrivDPDGC
}
