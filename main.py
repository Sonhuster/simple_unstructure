import numpy as np
import config as cf
import utils as uf
import solver as sol
import initialization as init

# Input definition
u_lid, v_lid = 1.0, 0.0
tol_inner = 1e-06
tol_outer = 1e-06
iter_outer = 2000
iter_mom = 1
iter_pp = 100
max_iteration = 1000

relax_uv = 0.8
relax_p = 0.15
noise = 0.00

if __name__ == '__main__':
    coordinates = uf.get_coordinate(4, 0.0)

    elem_coords = cf.get_elems(coordinates)
    n = len(np.unique(coordinates[:, 0])) - 1 # Number of cell along each dir
    n_v = len(coordinates)
    n_c = len(elem_coords)
    n_f = 2 * (n + 1) * n
    simplices = cf.get_simplices(coordinates, n_c, n)
    global_face = cf.get_global_face(n_v, n)
    neighbors = cf.get_neighbor_quad(n_c, n)
    convex_hull = cf.get_convex_hull(coordinates)
    coordinates = cf.get_noise(coordinates, noise)
    mesh = cf.BlockData2D(coordinates, simplices, neighbors, n_f)
    mesh.call_configuration(convex_hull, global_face)
    del coordinates, elem_coords, n, n_v, n_c, n_f, simplices, global_face, neighbors, convex_hull

    # coordinates = cf.get_noise(coordinates, noise)
    # tri = uf.create_tetrahedral_faces(coordinates)
    # tri.no_faces = uf.number_face_tri(tri)
    # mesh = cf.BlockData2D(tri.points, tri.simplices, tri.neighbors, tri.no_faces)
    # global_face = np.array([None])
    # mesh.call_configuration(tri.convex_hull, global_face)
    # del tri, coordinates


    # Node and face weighted coefficients
    fw = cf.cell_to_face_interpolation(mesh)  # [face]
    cw = cf.cell_to_node_interpolation(mesh)        # [node, cell]

    # Allocate memory && initialization
    var = init.Fluid(mesh)
    scx, scy, skewx, skewy, ap, res, anb = init.momentum_equation_arg(mesh)
    sc_p, ap_p, res_p, anb_p = init.possion_equation_arg(mesh)
    ucor, vcor, pcor, mdotfcor, pfcor = init.vel_correction_arg(mesh)

    cf.bc_face_lid_driven_cavity(mesh, var.ubc, var.vbc, u_lid, v_lid)              # Set ubc
    cf.bc_node_lid_driven_cavity(mesh, var.uv, var.vv, u_lid, v_lid)               # Set uv
    init.set_bc_face_from_local_to_global(mesh, var.ubc, var.vbc, var.uf, var.vf)    # Set uf
    uc_old, vc_old = var.uc, var.vc

    for iter_ in range(iter_outer):
        # Cal mom-coefficients
        # Gọi hai hàm
        ap, anb, scx, scy = sol.cal_momemtum_link_coeff(mesh, ap, scx, scy, var.mdotf, var.ubc, var.vbc)
        scx, scy = sol.cal_momentume_pressure_source(mesh, fw, scx, scy, var.pc, var.pf)                        # TODO: Calculate pressure face
        # result1 = (scx.copy(), scy.copy())
        # sol.cal_momentume_pressure_source1(mesh, fw, scx, scy, var.pc, var.pf)
        # result2 = (scx.copy(), scy.copy())
        # print(np.allclose(result1[0], result2[0]))  # So sánh ap
        # print(np.allclose(result1[1], result2[1]))  # So sánh scx
        var.uv, var.vv, var.pv = sol.cal_node_value(var.uv, var.vv, var.pv, var.uc, var.vc, var.pc, mesh, cw, u_lid, v_lid)
        skewx, skewy = sol.cal_momentum_skew_term(var.uv, var.vv, mesh)

        # Solve mom-equation
        var.uc = sol.solve_mom_eq(mesh, var.uc, ap, anb, scx, skewx, res, tag = "X")
        var.vc = sol.solve_mom_eq(mesh, var.vc, ap, anb, scy, skewy, res, tag = "Y")

        # Solve poison equation
        var.mdotf = sol.cal_massflow_face(mesh, var.uc, var.vc, var.pc, var.pf, ap, fw, var.mdotf)
        pcor = sol.solve_poison_eq(mesh, pcor, ap, anb_p, ap_p, sc_p, res_p, var.mdotf, fw)  # TODO: pressure link and source

        # Velocity and pressure corrected
        sol.corrected_cell_vel(mesh, ucor, vcor, pcor, var.uc, var.vc, pfcor, ap, fw)
        sol.corrected_massflux(mesh, ap, fw, mdotfcor, pcor, var.mdotf)
        var.pc, _ = sol.corrected_pressure(var.pc, var.pf, pcor, pfcor)  # TODO: should update pf?

        # Calculate residual
        error_u, error_v = sol.cal_outer_res(var.uc, var.vc, uc_old, vc_old)
        uc_old, vc_old = var.uc, var.vc
        print(f"Outer iteration {iter_} - residual (u, v) = ({error_u}, {error_v})")
        if error_u < 1e-05 and error_v < 1e-05:
            break

    var.uv, var.vv, var.pv = sol.cal_node_value(var.uv, var.vv, var.pv, var.uc, var.vc, var.pc, mesh, cw, u_lid, v_lid)
    uf.plot_vtk(var.uv, mesh, "Velocity U")
    uf.plot_vtk(var.vv, mesh, "Velocity V")
    uf.plot_vtk(np.sqrt(var.uv ** 2 + var.vv ** 2), mesh, "Velocity Magnitude")
    uf.plot_vtk(var.pv, mesh, "Pressure")
    print("Case has been done")

    # TODO: Investigate the remaining parameter (such as pressure relaxation)
    # TODO: Calculate the outer residual
