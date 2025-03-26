import numpy as np
import config as cf
import utils as uf
from main import tol_inner
from main import iter_mom
from main import iter_pp
from main import relax_uv
from main import relax_p

mu = 0.01
rho = 1.0

def cal_face_value(mesh, fw, var, varf):
    f2c, f_2_bf = uf.shorted_name(mesh.link, 'f2c', 'f_2_bf')
    for ifc in range(mesh.no_faces()):
        if f_2_bf[ifc] != -1:  # IF boundary face = continue
            continue
        c0 = f2c[ifc, 0]
        c1 = f2c[ifc, 1]
        varf[ifc] = fw[ifc] * var[c0] + (1 - fw[ifc]) * var[c1]

    return varf


def cal_node_value(uv, vv, pv, uc, vc, pc, mesh, cw, u_lid, v_lid):
    c2v = uf.shorted_name(mesh.link, 'c2v')[0]
    uv.fill(0), vv.fill(0), pv.fill(0)

    for ic in range(mesh.no_elems()):
        for local_node in range(mesh.no_local_faces()):
            iv = c2v[ic, local_node]
            if np.any(np.isin(iv, mesh.boundary_info.nodes)):   # IF boundary --nodes-- = continnue
                continue
            else:
                uv[iv] += uc[ic] * cw[iv, ic]
                vv[iv] += vc[ic] * cw[iv, ic]
                pv[iv] += pc[ic] * cw[iv, ic]

    cf.bc_node_lid_driven_cavity(mesh, uv, vv, u_lid, v_lid)
    return uv, vv, pv


def cal_momemtum_link_coeff(mesh, ap, scx, scy, anb, mdotf, ubc, vbc):
    global mu
    area, delta, snsign = uf.shorted_name(mesh.global_faces, 'area', 'delta', 'snsign')
    c2f, f_2_bf = uf.shorted_name(mesh.link, 'c2f', 'f_2_bf')
    ap.fill(0.0), scx.fill(0.0), scy.fill(0.0)

    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = c2f[ic, local_face]
            mf = mdotf[ifc] * snsign[ic, local_face]
            if f_2_bf[ifc] == -1:
                ap[ic] += mu * area[ifc] / delta[ifc] + 0.5 * (np.abs(mf) + mf)
                anb[ic, local_face] = - mu * area[ifc] / delta[ifc] - 0.5 * (np.abs(mf) - mf)
            else:
                ifb = f_2_bf[ifc] # Local boundary face idx
                ap[ic] += mu * area[ifc] / delta[ifc]
                anb[ic, local_face] = 0.0
                scx[ic] += ubc[ifb] * mu * area[ifc] / delta[ifc] - mf * ubc[ifb]
                scy[ic] += vbc[ifb] * mu * area[ifc] / delta[ifc] - mf * vbc[ifb]


def cal_momentume_pressure_source(mesh, fw, scx, scy, pc, pf):
    area, sn, snsign = uf.shorted_name(mesh.global_faces, 'area', 'sn', 'snsign')
    c2f = uf.shorted_name(mesh.link, 'c2f')[0]
    # Calculate pressure face value
    pf = cal_face_value(mesh, fw, pc, pf)

    # Calculate pressure source for momentum eq.
    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = c2f[ic, local_face]
            scx[ic] -= pf[ifc] * sn[ifc, 0] * snsign[ic, local_face] * area[ifc]
            scy[ic] -= pf[ifc] * sn[ifc, 1] * snsign[ic, local_face] * area[ifc]


def cal_momentum_skew_term(skewx, skewy, uv, vv, mesh):
    centroid = uf.shorted_name(mesh.elems, 'centroid')[0]
    st, snsign, delta = uf.shorted_name(mesh.global_faces, 'st', 'snsign', 'delta')
    c2f, f2c, f2v, f_2_bf = uf.shorted_name(mesh.link, 'c2f', 'f2c', 'f2v', 'f_2_bf')
    skewx.fill(0.0), skewy.fill(0.0)
    for ic in range(mesh.no_elems()):
        sumfx, sumfy = 0.0, 0.0
        for local_face in range(mesh.no_local_faces()):
            ifc = c2f[ic, local_face]
            if f_2_bf[ifc] != -1:                     # IF boundary face = continue
                continue
            c0 = f2c[ifc, 0]
            c1 = f2c[ifc, 1]
            v0 = f2v[ifc, 0]
            v1 = f2v[ifc, 1]
            dx1 = centroid[c1, 0] - centroid[c0, 0]
            dy1 = centroid[c1, 1] - centroid[c0, 1]
            tdotl = st[ifc, 0] * dx1 + st[ifc, 1] * dy1
            sumfx += tdotl * (uv[v1] - uv[v0]) * snsign[ic, local_face] / delta[ifc]
            sumfy += tdotl * (vv[v1] - vv[v0]) * snsign[ic, local_face] / delta[ifc]

        skewx[ic] = sumfx * mu
        skewy[ic] = sumfy * mu


def solve_x_mom(mesh, uc_, ap_, anb_, scx_, skewx_, res_):
    def cal_residual(mesh, uc, ap, anb, scx, skewx, res, res2x, iter_):
        res.fill(0.0)
        sumr = 0.0      # residual summation
        for ic in range(mesh.no_elems()):
            sumf = 0.0
            for local_neighbor in range(mesh.no_local_faces()):   # Summing over all neighborhoods
                ifc = mesh.link.c2f[ic, local_neighbor]
                if mesh.global_faces.snsign[ic, local_neighbor] == 1:
                    icn = mesh.link.f2c[ifc, 1]
                else:
                    icn = mesh.link.f2c[ifc, 0]

                sumf += anb[ic, local_neighbor] * uc[icn]

            res[ic] = scx[ic] + skewx[ic] - ap[ic] * uc[ic] - sumf
            sumr += res[ic] ** 2

        res2 = np.sqrt(np.maximum(0.0, sumr))

        if iter_ == 0:
            res2x = res2

        return res2, res2x

    def cal_gauss_seidel_loop(uc, ap, res, rin_uv):
        utild = res / (ap * (1.0 + rin_uv))

        assert not np.any(np.isnan(uc + utild)), "X momentum - NAN value"
        assert not np.any(np.isinf(uc + utild)), "X momentum - INF value"
        return uc + utild

    res2x = 0.0
    for iter_ in range(iter_mom):
        res2, res2x = cal_residual(mesh, uc_, ap_, anb_, scx_, skewx_, res_, res2x, iter_)
        uc_ = cal_gauss_seidel_loop(uc_, ap_, res_, rin_uv = 0.1)

        # print(f"x-mom: iter {iter_} residual {res2}")
        if res2x == 0.0:
            print(f"\tx-mom converged at {iter_} iteration")
            break
        if res2/res2x < tol_inner:
            print(f"\tx-mom converged at {iter_} iteration")
            break

    return uc_


def solve_y_mom(mesh, vc_, ap_, anb_, scy_, skewy_, res_):
    def cal_residual(mesh, vc, ap, anb, scy, skewy, res, res2y, iter_) :
        res.fill(0.0)
        sumr = 0.0
        for ic in range(mesh.no_elems()):
            sumf = 0.0
            for local_neighbor in range(mesh.no_local_faces()):   # Summing over all neighborhoods
                ifc = mesh.link.c2f[ic, local_neighbor]
                if mesh.global_faces.snsign[ic, local_neighbor] == 1:
                    icn = mesh.link.f2c[ifc, 1]
                else:
                    icn = mesh.link.f2c[ifc, 0]

                sumf += anb[ic, local_neighbor] * vc[icn]

            res[ic] = scy[ic] + skewy[ic] - ap[ic] * vc[ic] - sumf
            sumr += res[ic] ** 2

        res2 = np.sqrt(np.maximum(0.0, sumr))

        if iter_ == 0:
            res2y = res2

        return res2, res2y

    def cal_gauss_seidel_loop(vc, ap, res, rin_uv):  # correction
        vtild = res / (ap * (1.0 + rin_uv))

        assert not np.any(np.isnan(vc + vtild)), "Y momentum - NAN value"
        assert not np.any(np.isinf(vc + vtild)), "Y momentum - INF value"
        return vc + vtild

    res2y = 0.0
    for iter_ in range(iter_mom):
        res2, res2y = cal_residual(mesh, vc_, ap_, anb_, scy_, skewy_, res_, res2y, iter_)
        vc_ = cal_gauss_seidel_loop(vc_, ap_, res_, rin_uv = 0.1)

        # print(f"y-mom: iter {iter_} residual {res2}")
        if res2y == 0:
            print(f"\ty-mom converged at {iter_} iteration")
            break
        if res2/res2y <= tol_inner:
            print(f"\ty-mom converged at {iter_} iteration")
            break

    return vc_


def cal_massflow_face(mesh, uc, vc, pc, pf, ap, fw, mdotf):
    vol = mesh.elems.volume
    for ifc in range(mesh.no_faces()):
        if mesh.link.f_2_bf[ifc] != -1:                     # IF boundary face = continue
            continue
        c0 = mesh.link.f2c[ifc, 0]
        c1 = mesh.link.f2c[ifc, 1]
        velf_x = fw[ifc] * uc[c0] + (1.0 - fw[ifc]) * uc[c1]
        velf_y = fw[ifc] * vc[c0] + (1.0 - fw[ifc]) * vc[c1]
        velf_i = velf_x * mesh.global_faces.sn[ifc, 0] + velf_y * mesh.global_faces.sn[ifc, 1]

        V0dp0_x = 0.0
        V0dp0_y = 0.0
        for local_face in range(mesh.no_local_faces()):
            iff = mesh.link.c2f[c0, local_face]
            V0dp0_x += pf[iff] + mesh.global_faces.area[iff] * mesh.global_faces.sn[iff, 0] * mesh.global_faces.snsign[c0, local_face]
            V0dp0_y += pf[iff] + mesh.global_faces.area[iff] * mesh.global_faces.sn[iff, 1] * mesh.global_faces.snsign[c0, local_face]

        V1dp1_x = 0.0
        V1dp1_y = 0.0
        for local_face in range(mesh.no_local_faces()):
            iff = mesh.link.c2f[c1, local_face]
            V1dp1_x += pf[iff] + mesh.global_faces.area[iff] * mesh.global_faces.sn[iff, 0] * mesh.global_faces.snsign[c1, local_face]
            V1dp1_y += pf[iff] + mesh.global_faces.area[iff] * mesh.global_faces.sn[iff, 1] * mesh.global_faces.snsign[c1, local_face]

        velf_x = fw[ifc] * V0dp0_x / ap[c0] + (1.0 -fw[ifc]) * V1dp1_x / ap[c1]
        velf_y = fw[ifc] * V0dp0_y / ap[c0] + (1.0 -fw[ifc]) * V1dp1_y / ap[c1]
        velf_p = velf_x * mesh.global_faces.sn[ifc, 0] + velf_y * mesh.global_faces.sn[ifc, 1]
        vdotn = velf_i + velf_p \
                - (fw[ifc] * vol[c0] / ap[c0] + (1.0 - fw[ifc]) * vol[c1] / ap[c1]) \
                * (pc[c1] - pc[c0]) / mesh.global_faces.delta[ifc]
        mdotf[ifc] = vdotn * rho * mesh.global_faces.area[ifc]  # mass flow rate in kg/s

        return mdotf


def cal_source_mass_imbalance(mesh, sc_p, mdotf):    # Calculate source due to mass imbalance
    sc_p.fill(0.0)
    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = mesh.link.c2f[ic, local_face]
            sc_p[ic] -= mdotf[ifc] * mesh.global_faces.snsign[ic, local_face]

    return sc_p

def cal_pressure_link_coeff(mesh, ap, ap_p, anb_p, fw):
    vol = mesh.elems.volume
    ap_p.fill(0.0), anb_p.fill(0.0)
    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = mesh.link.c2f[ic, local_face]
            if mesh.link.f_2_bf[ifc] != -1:  # IF boundary face = continue
                continue
            if mesh.global_faces.snsign[ic, local_face] == 1:
                icn = mesh.link.f2c[ifc, 1]
            else:
                icn = mesh.link.f2c[ifc, 0]
            ap_p[ic] += (fw[ifc] * vol[ic] / ap[ic] + (1.0 - fw[ifc]) * vol[icn] / ap[icn]) \
                            * rho * mesh.global_faces.area[ifc] / mesh.global_faces.delta[ifc]
            anb_p[ic, local_face] = - (fw[ifc] * vol[ic] / ap[ic] + (1.0 - fw[ifc]) * vol[icn] / ap[icn]) \
                            * rho * mesh.global_faces.area[ifc] / mesh.global_faces.delta[ifc]


def solve_poison_eq(mesh, pcor_, ap_, anb_p_, ap_p_, sc_p_, res_p_):
    def cal_gauss_seidel_loop(mesh, pcor, anb_p, ap_p, sc_p):
        for ic in range(mesh.no_elems()):
            sumf = 0.0
            for local_face in range(mesh.no_local_faces()):
                ifc = mesh.link.c2f[ic, local_face]
                if mesh.global_faces.snsign[ic, local_face] == 1:
                    icn = mesh.link.f2c[ifc, 1]
                else:
                    icn = mesh.link.f2c[ifc, 0]
                sumf += anb_p[ic, local_face] * pcor[icn]
            pcor[ic] = (sc_p[ic] - sumf) / ap_p[ic]

        return pcor

    def cal_residual(pcor, pcor_old, res2p, iter_):
        res = np.linalg.norm(pcor - pcor_old)
        res2 = np.sqrt(np.maximum(0.0, res))
        if iter_ == 0:
            res2p = res2

        return res2, res2p

    pcor_.fill(0.0)
    res2p = 0.0
    for iter_ in range(iter_pp):
        pcor_old = pcor_.copy()
        pcor_ = cal_gauss_seidel_loop(mesh, pcor_, anb_p_, ap_p_, sc_p_)
        res2, res2p = cal_residual(pcor_, pcor_old, res2p, iter_)

        # print(f"poison eq: iter {iter_} residual {res2}")
        if res2p == 0:
            print(f"\tpoison eq converged at {iter_} iteration")
            break
        if res2 / res2p < tol_inner:
            print(f"\tpoison eq converged at {iter_} iteration")
            break
    return pcor_


def corrected_cell_vel(mesh, ucor, vcor, pcor, uc, vc, pfcor, ap, fw):
    # Calculate correction pressure face value
    pfcor = cal_face_value(mesh, fw, pcor, pfcor)

    ucor.fill(0.0), vcor.fill(0.0)
    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = mesh.link.c2f[ic, local_face]
            ucor[ic] += pfcor[ifc] * mesh.global_faces.sn[ifc, 0] * mesh.global_faces.snsign[ic, local_face] * mesh.global_faces.area[ifc]
            vcor[ic] += pfcor[ifc] * mesh.global_faces.sn[ifc, 1] * mesh.global_faces.snsign[ic, local_face] * mesh.global_faces.area[ifc]

        ucor[ic] = - ucor[ic] / ap[ic]
        vcor[ic] = - vcor[ic] / ap[ic]
        uc[ic] += relax_uv * ucor[ic]
        vc[ic] += relax_uv * vcor[ic]


def corrected_massflux(mesh, ap, fw, mdotfcor, pcor, mdotf):
    vol = mesh.elems.volume
    for ifc in range(mesh.no_faces()):
        if mesh.link.f_2_bf[ifc] != -1:  # IF boundary face = continue
            continue
        c0 = mesh.link.f2c[ifc, 0]
        c1 = mesh.link.f2c[ifc, 1]
        coeff = fw[ifc] * vol[c0] / ap[c0] + (1.0 - fw[ifc]) * vol[c1] / ap[c1]
        mdotfcor[ifc] = rho * coeff * mesh.global_faces.area[ifc] * (pcor[c0] - pcor[c1]) / mesh.global_faces.delta[ifc]
        mdotf[ifc] = mdotf[ifc] + relax_uv * mdotfcor[ifc]


def corrected_pressure(pc, pf, pcor, pfcor):
    pc = pc + relax_p * pcor
    # pf = pf + relax_p * pfcor
    return pc, pf


def cal_outer_res(uc, vc, uc_old, vc_old):
    error_u = np.sum(np.abs(uc - uc_old)) / np.sum(np.abs(uc))
    error_v = np.sum(np.abs(vc - vc_old)) / np.sum(np.abs(vc))

    return error_u, error_v


