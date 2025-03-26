import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import utils as uf

# --------------------------Initialization------------------------------ #
class Fluid:
    def __init__(self, mesh):
        n_cells = mesh.no_elems()
        n_faces = mesh.no_faces()
        n_nodes = mesh.no_nodes()

        self.uc = np.zeros(n_cells)
        self.vc = np.zeros(n_cells)
        self.pc = np.ones(n_cells) * 0

        self.uf = np.zeros(n_faces)
        self.vf = np.zeros(n_faces)
        self.pf = np.ones(n_faces) * 0
        self.mdotf = np.zeros(n_faces)

        self.uv = np.zeros(n_nodes)
        self.vv = np.zeros(n_nodes)
        self.pv = np.ones(n_nodes) * 0

        self.ubc = np.zeros(len(mesh.boundary_info.faces))
        self.vbc = np.zeros(len(mesh.boundary_info.faces))

def momentum_equation_arg(mesh):
    n_cells = mesh.no_elems()
    n_local_faces = mesh.no_local_faces()

    scx = np.zeros(n_cells)
    scy = np.zeros(n_cells)
    skewx = np.zeros(n_cells)
    skewy = np.zeros(n_cells)
    ap = np.zeros(n_cells)
    res = np.zeros(n_cells)
    anb = np.zeros((n_cells, n_local_faces))
    return scx, scy, skewx, skewy, ap, res, anb

def possion_equation_arg(mesh):
    n_cells = mesh.no_elems()
    n_local_faces = mesh.no_local_faces()

    sc_p = np.zeros(n_cells)
    ap_p = np.zeros(n_cells)
    res_p = np.zeros(n_cells)
    anb_p = np.zeros((n_cells, n_local_faces))

    return sc_p, ap_p, res_p, anb_p

def vel_correction_arg(mesh):
    n_cells = mesh.no_elems()
    n_faces = mesh.no_faces()

    ucor = np.zeros(n_cells)
    vcor = np.zeros(n_cells)
    pcor = np.zeros(n_cells)
    mdotfcor = np.zeros(n_faces)
    pfcor = np.zeros(n_faces)

    return ucor, vcor, pcor, mdotfcor, pfcor

# --------------------------Boundary condition------------------------------ #
def set_bc_face_from_local_to_global(mesh, ubc, vbc, uf, vf):
    for ifb in range(len(mesh.boundary_info.faces)):
        ifc = mesh.link.bf_2_f[ifb]
        uf[ifc] = ubc[ifb]
        vf[ifc] = vbc[ifb]
