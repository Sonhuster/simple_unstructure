import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import utils as uf

def cell_to_face_interpolation(mesh):
    wf = np.zeros(mesh.no_faces())
    for ifc in range(mesh.no_faces()):
        ic0 = mesh.link.f2c[ifc, 0]
        ic1 = mesh.link.f2c[ifc, 1]
        d0 = np.linalg.norm(mesh.elems.centroid[ic0] - mesh.global_faces.centroid[ifc])
        d1 = np.linalg.norm(mesh.elems.centroid[ic1] - mesh.global_faces.centroid[ifc])
        wf[ifc] = (1/d0) / ((1/d0) + (1/d1))

    return wf

def cell_to_node_interpolation(mesh):
    wv = np.zeros((mesh.no_nodes(), mesh.no_elems()))
    for inc in range(mesh.no_nodes()):
        contain_elems = np.where(np.isin(mesh.elems.define, inc) == True)[0]
        node_coord = mesh.coords[inc]
        elem_coords = mesh.elems.centroid[contain_elems]
        o_distance = 1 / np.linalg.norm(node_coord - elem_coords, axis = 1)
        wv_ = o_distance / (np.sum(o_distance))
        wv[inc, contain_elems] = wv_

    return wv

class Face:
    def __init__(self, number_of_faces):
        self.define = np.array([None])
        self.centroid = np.zeros(number_of_faces)
        self.area = np.array([None])
        self.sn = np.array([None])
        self.st = np.array([None])
        self.snsign = np.array([None])
        self.delta = np.array([None])

class Element:
    def __init__(self, elems):
        self.define = elems
        self.centroid = np.array([None])
        self.volume = np.array([None])

class BoundaryInfo:
    def __init__(self):
        self.faces = np.array([None])
        self.nodes = np.array([None])
        self.face_patches = {}
        self.node_patches = {}

class Connectivity:
    def __init__(self, no_cells, no_faces, no_local_faces):
        self.c2f = np.zeros((no_cells, no_local_faces), dtype = int)    # [Global cell, local face] =  Global face
        self.c2v = np.zeros((no_cells, no_local_faces), dtype = int)     # [Global cell, local node] =  Global node
        self.f2c = np.zeros((no_faces, 2), dtype = int)     # [Global face, local cell] =  Global cell
        self.f2v = np.zeros((no_cells, 2), dtype = int)     # [Global face, local node] =  Global node
        self.bf_2_f = np.array([None])
        self.f_2_bf = np.array([None])

class BlockData2D:
    def __init__(self, coords, elems, neighbors, no_faces):
        self.coords = coords
        self.elems = Element(elems)
        self.global_faces = Face(no_faces)
        self.cell_mapping = np.array([None])
        self.neighbors = neighbors
        self.boundary_info = BoundaryInfo()
        self.link = Connectivity(self.no_elems(), self.no_faces(), self.no_local_faces())

    def no_elems(self):
        return len(self.elems.define)

    def no_nodes(self):
        return len(self.coords)

    def no_faces(self):
        return len(self.global_faces.centroid)

    def no_local_faces(self):
        return self.elems.define.shape[1]

    def elems_centroid(self):
        centroid_coords = self.coords[self.elems.define]
        self.elems.centroid = np.mean(centroid_coords, axis = 1)

    def face_centroid(self):
        face_coords = self.coords[self.global_faces.define]
        self.global_faces.centroid = np.mean(face_coords, axis=1)

    def get_face_area(self):
        node_coords = self.coords[self.global_faces.define]
        self.global_faces.area =  np.linalg.norm(node_coords[:, 0] - node_coords[:, 1], axis = 1)

    def get_elem_volume(self):
        x = self.coords[self.elems.define][:, :, 0]
        y = self.coords[self.elems.define][:, :, 1]

        if self.no_local_faces() == 3:
            self.elems.volume = 0.5 * np.abs(
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
            )
        elif self.no_local_faces() == 4:
            self.elems.volume = (0.5 * np.abs(
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
            ) +
                                 0.5 * np.abs(
                x[:, 0] * (y[:, 2] - y[:, 3]) +
                x[:, 2] * (y[:, 3] - y[:, 0]) +
                x[:, 3] * (y[:, 0] - y[:, 2])
            ))


    # Get global face index + face_to_cell_connectivity + face_to_node_connectivity
    def global_face_numbering_tri(self):
        """
        Algorithm: Face-th is the opposite to the deleted node-th.
        :param : Simplices matrix of Delaunay triangles.
        :return: Global Face index, Face-to-Cell connectivity
        """
        global_faces_temp = set()

        global_faces = {}
        cell_mapping = {}
        global_face_id = 0
        cell_id = 0

        for simplex in self.elems.define:
            for i in range(self.no_local_faces()):
                face = (simplex[(i + 1) % 3], simplex[(i + 2) % 3])
                sorted_face = tuple(sorted(face))

                if sorted_face not in global_faces_temp:
                    global_faces_temp.add(sorted_face)
                    global_faces[face] = global_face_id
                    cell_mapping[tuple([global_face_id, 0])] = cell_id

                    global_face_id += 1
                else:
                    if face in global_faces:
                        cell_mapping[tuple([global_faces[face], 1])] = cell_id
                    else:
                        cell_mapping[tuple([global_faces[face[::-1]], 1])] = cell_id
            cell_id += 1

        # From dict to array
        cell_mapping_ = np.full((self.no_faces(), 2), -1)
        for (i, j), value in cell_mapping.items():
            cell_mapping_[i, j] = value

        for i in range(self.no_faces()):
            if cell_mapping_[i, 1] == -1:
                cell_mapping_[i, 1] = cell_mapping_[i, 0]

        global_face_ = np.zeros((self.no_faces(), 2), dtype = int)
        dem = 0
        for i, j in global_faces.keys():
            global_face_[dem, 0] = i
            global_face_[dem, 1] = j
            dem += 1

        self.global_faces.define = global_face_
        self.link.f2c = cell_mapping_
        self.link.f2n = global_face_

    def global_face_numbering_quad(self, global_face):
        """
        Algorithm: Face-th is the opposite to the deleted node-th.
        :param : Simplices matrix of Delaunay triangles.
        :return: Global Face index, Face-to-Cell connectivity
        """
        self.global_faces.define = global_face

        cell_mapping = {}

        for idx in range(self.no_faces()):
            face = self.global_faces.define[idx]

            cell_idx = tuple(sorted(np.where(np.sum(np.isin(self.elems.define, face), axis=1) == 2)[0]))
            cell_idx = cell_idx if len(cell_idx) > 1 else (cell_idx[0], -1)
            cell_mapping[idx] = cell_idx

        # From dict to array
        cell_mapping_ = np.full((self.no_faces(), 2), -1)
        for value, (i, j) in cell_mapping.items():
            cell_mapping_[value] = np.array([i, j])

        for i in range(self.no_faces()):
            if cell_mapping_[i, 1] == -1:
                cell_mapping_[i, 1] = cell_mapping_[i, 0]

        self.link.f2c = cell_mapping_
        self.link.f2n = self.global_faces.define


    # From global cell + local face, points out global face
    def cal_l_cell_to_face(self):
        simplex = self.elems.define
        local_face_index = np.arange(self.no_local_faces())
        if self.no_local_faces() == 3:
            for ic in range(self.no_elems()):
                for face_order in local_face_index:
                    face = np.delete(simplex[ic], face_order)
                    global_face_idx = np.where(np.sum(np.isin(self.global_faces.define, face), axis=1) == 2)[0][0]
                    self.link.c2f[ic, face_order] = global_face_idx
        else:
            for ic in range(self.no_elems()):
                for face_order in local_face_index:
                    face = np.array([simplex[ic, face_order % self.no_local_faces()],
                                     simplex[ic, (face_order + 1) % self.no_local_faces()]])
                    global_face_idx = np.where(np.sum(np.isin(self.global_faces.define, face), axis=1) == 2)[0][0]
                    self.link.c2f[ic, face_order] = global_face_idx

    # Return local boundary face if bface, or -1 if interior face
    def cal_link_boundary_face(self):
        f_to_bf = np.full(self.no_faces(), -1)
        global_face_idx = np.arange(self.no_faces())
        indices_in_b = np.where(np.isin(global_face_idx, self.boundary_info.faces))[0]
        for idx in indices_in_b:
            f_to_bf[idx] = np.where(self.boundary_info.faces == global_face_idx[idx])[0][0]

        self.link.f_2_bf = f_to_bf
        self.link.bf_2_f = self.boundary_info.faces

    # Get the indices where faces are on boundary.
    def domain_boundary(self, convex_hull):
        self.boundary_info.nodes = np.unique(convex_hull)
        for i in range(10):
            temp1 = convex_hull + i
            check1 = temp1[:, 0] * temp1[:, 1]
            temp2 = self.global_faces.define + i
            check2 = temp2[:, 0] * temp2[:, 1]
            if len(check1) == len(np.unique(check1)) and len(check2) == len(np.unique(check2)):
                break
            else:
                continue

        self.boundary_info.faces = np.where(np.isin(check2, check1) == True)[0]

    # # Get local outward normal vector of each element
    # def get_normal_face(self):
    #     local_normal = np.zeros((self.no_elems(), self.no_local_faces(), 2))
    #     for ic in range(self.no_elems()):
    #         for ifc in range(self.no_local_faces()):
    #             global_if = self.L_cell_to_face(ic, ifc)
    #             global_nf = self.global_faces.define[global_if]
    #             coords = self.coords[global_nf]
    #             vector = coords[1] - coords[0]
    #             vector_centroid = coords[0] - self.elems.centroid[ic]
    #             # Because Delaunay-triangle has been defined to counter-clockwise, then take normal vector rotating 90 degrees - clockwise to the original vector
    #             local_normal_ = uf.normalized_vector(np.array([vector[1], -vector[0]]))
    #             local_normal[ic, ifc] = local_normal_ if np.dot(vector_centroid, local_normal_) > 0 else - local_normal_
    #
    #     self.global_faces.sn = local_normal

    # Get local outward normal vector defined from cell 1 to 2
    def get_face_sn_st(self):
        normal = np.zeros((self.no_faces(), 2))
        tangent = np.zeros((self.no_faces(), 2))
        for ifc in range(self.no_faces()):
            coords = self.coords[self.global_faces.define[ifc]]
            vector = coords[1] - coords[0]
            centroid = self.elems.centroid[self.link.f2c[ifc, 0]]
            vector_centroid = coords[0] - centroid
            normal_ = np.array([vector[1], -vector[0]])

            if np.dot(vector_centroid, normal_) > 0:
                normal[ifc] = uf.normalized_vector(normal_)
                tangent[ifc] = uf.normalized_vector(vector)
            else:
                normal[ifc] = - uf.normalized_vector(normal_)
                tangent[ifc] = - uf.normalized_vector(vector)

        self.global_faces.sn = normal
        self.global_faces.st = tangent

    def get_normal_face_sign(self):
        local_normal_sign = np.zeros((self.no_elems(), self.no_local_faces()))
        for ic in range(self.no_elems()):
            for ifc in range(self.no_local_faces()):
                ifc_global = self.link.c2f[ic, ifc]
                ic_temp = self.link.f2c[ifc_global, 0]
                if ic == ic_temp:
                    local_normal_sign[ic, ifc] = 1
                else:
                    local_normal_sign[ic, ifc] = -1

        self.global_faces.snsign = local_normal_sign

    # Delta = distance between two cell in normal direction
    def delta_distance_cal(self):
        vec_l = np.zeros((self.no_faces(), 2))
        for ifc in range(self.no_faces()):
            ic0 = self.link.f2c[ifc, 0]
            ic1 = self.link.f2c[ifc, 1]
            if self.link.f_2_bf[ifc] == -1:
                vec_l[ifc] = self.elems.centroid[ic1] - self.elems.centroid[ic0]
            else:
                vec_l[ifc] = self.global_faces.centroid[ifc] - self.elems.centroid[ic0]

        self.global_faces.delta = np.abs(np.sum(self.global_faces.sn * vec_l, axis=1))

    def set_patches(self):
        # Set face patches
        boundary_face_centroids = self.global_faces.centroid[self.boundary_info.faces]
        self.boundary_info.face_patches["bot walls"] = self.boundary_info.faces[boundary_face_centroids[:, 1] < 1e-03]
        self.boundary_info.face_patches["zero walls"] = self.boundary_info.faces[boundary_face_centroids[:, 1] > 1e-03]

        boundary_nodes =  self.coords[self.boundary_info.nodes]
        self.boundary_info.node_patches["bot nodes"] = self.boundary_info.nodes[boundary_nodes[:, 1] < 1e-03]
        self.boundary_info.node_patches["zero nodes"] = self.boundary_info.nodes[boundary_nodes[:, 1] > 1e-03]

    def call_configuration(self, convex_hull, global_face):
        self.elems_centroid()
        self.get_elem_volume()
        self.link.c2v = self.elems.define
        if self.no_local_faces() == 3:
            self.global_face_numbering_tri()
        else:
            self.global_face_numbering_quad(global_face)
        self.face_centroid()
        self.get_face_area()
        self.cal_l_cell_to_face()
        self.link.f2v = self.global_faces.define
        self.domain_boundary(convex_hull)
        self.cal_link_boundary_face()
        self.get_face_sn_st()
        self.get_normal_face_sign()
        self.delta_distance_cal()
        self.set_patches()

# --------------------------Boundary condition------------------------------ #
def bc_face_lid_driven_cavity(mesh, ubc, vbc, u_lid, v_lid):
    for ifc in mesh.boundary_info.face_patches['bot walls']:
        ifb = mesh.link.f_2_bf[ifc]
        ubc[ifb] = u_lid
        vbc[ifb] = v_lid


def bc_node_lid_driven_cavity(mesh, uv, vv, u_lid, v_lid):
    assert len(uv) == mesh.no_nodes()
    assert len(vv) == mesh.no_nodes()

    uv[mesh.boundary_info.node_patches['bot nodes']] = u_lid
    uv[mesh.boundary_info.node_patches['zero nodes']] = 0.0
    vv[mesh.boundary_info.node_patches['bot nodes']] = v_lid
    vv[mesh.boundary_info.node_patches['zero nodes']] = 0.0

# ---------------------------------------------------------------------------- #
def get_elems(coords):
    mean_x = 0.5 * (np.unique(coords[:, 0])[0:-1] + np.unique(coords[:, 0])[1:])
    mean_y = 0.5 * (np.unique(coords[:, 1])[0:-1] + np.unique(coords[:, 1])[1:])
    X, Y = np.meshgrid(mean_x, mean_y)
    elem_coords = np.dstack((X, Y)).reshape(-1, 2)
    return elem_coords

def get_simplices(coords, n_c, n):
    arr = np.tile(np.arange(n_c), (4, 1)).T
    arr_ = np.tile(np.array([0, 1, n + 2, n + 1]), (n_c, 1))
    arr__ = np.tile(np.arange(n), (n, 1)).T.ravel().reshape(-1, 1)
    return arr + arr_ + arr__

def get_global_face(n_v, n):
    node_idx = np.arange(n_v).reshape(-1, n + 1)
    row_face = np.zeros((n + 1, n, 2), dtype = int)
    for i in range(len(row_face)):
        row_face[i] = np.column_stack((node_idx[i, :-1], node_idx[i, 1:]))
    row_face = row_face.reshape(-1, 2)

    col_face = np.zeros((n, n + 1, 2), dtype=int)
    for i in range(len(col_face)):
        col_face[i] = np.column_stack((node_idx[i], node_idx[i + 1]))
    col_face = col_face.reshape(-1, 2)

    return np.vstack((row_face, col_face))

def get_neighbor_quad(n_c, n):
    neighbor = np.zeros((n_c, 4), dtype = int)
    for i in range(n_c):
        neighbor[i] = np.array([i - n, i + 1, i + n, i - 1])
        if i % n == 0:
            neighbor[i, 3] = -1
        if (i + 1) % n == 0:
            neighbor[i, 1] = -1
        if (i + n) >= n_c:
            neighbor[i, 2] = -1
        if (i - n) < 0:
            neighbor[i, 0] = -1

    return neighbor

def get_convex_hull(coords):
    # Western
    west = np.where(coords[:, 0] < 1e-03)[0][::-1]
    southern = np.where(coords[:, 1] < 1e-03)[0]
    eastern = np.where(coords[:, 0] > np.max(coords[:, 0]) - 1e-03)[0]
    northen = np.where(coords[:, 1] > np.max(coords[:, 1]) - 1e-03)[0][::-1]

    convex_hull = np.concatenate((west, southern[1:], eastern[1:], northen[1:]))
    return np.column_stack((convex_hull[:-1], convex_hull[1:]))