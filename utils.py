import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv

def shorted_name(mesh, *attrs):
    return [getattr(mesh, attr) for attr in attrs]


def normalized_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("! Vector is zero")

    return vector / norm

def get_total_flux(*vars):
    result = []
    for var in vars:
        result.append(np.sum(var, axis=1))

    return tuple(result)


def get_coordinate(number, skewness):
    x = np.linspace(0, 0.9, number)
    y = np.linspace(0, 0.9, number)
    z = np.linspace(0, 0, number)

    if len(np.unique(z)) == 1:
        X, Y = np.meshgrid(x, y)
        coords = np.vstack([X.ravel(), Y.ravel()]).T
    else:
        X, Y, Z = np.meshgrid(x, y, z)
        coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # # Add random distortion for interior points
    # for i in range(len(coords)):
    #     # Skip boundary points (x = 0 or x = 0.9, y = 0 or y = 0.9)
    #     if 0 < coords[i, 0] < 0.9 and 0 < coords[i, 1] < 0.9:
    #         # Add random noise to interior points
    #         coords[i, 0] += np.random.uniform(-skewness, skewness)  # Small perturbation for x
    #         coords[i, 1] += np.random.uniform(-skewness, skewness)  # Small perturbation for y

    return coords


def create_tetrahedral_faces(coordinates):
    tri = Delaunay(coordinates)
    return tri

def number_face_tri(tri):
    num_faces = 0
    for simplex_index, neighbors in enumerate(tri.neighbors):
        for neighbor in neighbors:
            if neighbor == -1:
                num_faces += 1
            elif neighbor > simplex_index:
                num_faces += 1

    return num_faces

def plot_tetrahedral_surface(coordinates, faces):
    fig = plt.figure()

    if coordinates.shape[1] == 2:
        ax = fig.add_subplot(111)
        plt.triplot(coordinates[:, 0], coordinates[:, 1], faces)
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], color='r')
        for tetra in faces:
            for i in range(4):
                triangle = coordinates[np.delete(tetra, i)]  # Get triangle from 3 nodes of a tetrahedron
                ax.add_collection3d(Poly3DCollection([triangle], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()

def delete_redundant_vars(*var_names):
    for var_name in var_names:
        if var_name in globals():
            del globals()[var_name]
        elif var_name in locals():
            del locals()[var_name]

def plot_vtk(phi_node, mesh, field):
    if mesh.no_local_faces() == 3:
        nodes = np.hstack((mesh.coords, np.zeros(mesh.no_nodes()).reshape(-1, 1)))
        cells = np.hstack([[3] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.TRIANGLE] * mesh.no_elems()
        grid = pv.UnstructuredGrid(cells, celltypes, nodes)
        temperature = phi_node
        grid.point_data[field] = temperature

        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, scalars=field, cmap="jet")
        plotter.show_axes()

        plotter.add_points(np.array([0, 0, 0]), color="black", point_size=10, render_points_as_spheres=True)
        plotter.add_point_labels([[0, 0, 0]], ['Origin'], point_size=20, text_color="black")
    else:
        nodes = np.hstack((mesh.coords, np.zeros(mesh.no_nodes()).reshape(-1, 1)))
        cells = np.hstack([[4] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.QUAD] * mesh.no_elems()
        grid = pv.UnstructuredGrid(cells, celltypes, nodes)
        temperature = phi_node
        grid.point_data[field] = temperature
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, scalars=field, cmap="jet")
        plotter.show_axes()

        plotter.add_points(np.array([0, 0, 0]), color="black", point_size=10, render_points_as_spheres=True)
        plotter.add_point_labels([[0, 0, 0]], ['Origin'], point_size=20, text_color="black")
    plotter.isometric_view_interactive()
    plotter.show()