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
    (var_,) = mesh.var_face_wise(var,)
    weighted_coeff = np.vstack((fw, 1 - fw)).T
    maskf_bc = f_2_bf >= 0
    maskf_in = ~ maskf_bc

    # Keep face values on BCs
    varf_bc = varf * maskf_bc
    varf_in = np.sum(var_ * weighted_coeff, axis=1) * maskf_in

    return varf_bc + varf_in


def cal_node_value(uv, vv, pv, uc, vc, pc, mesh, cw, u_lid, v_lid):
    uv = np.sum(uc * cw, axis=1)
    vv = np.sum(vc * cw, axis=1)
    pv = np.sum(pc * cw, axis=1)

    cf.bc_node_lid_driven_cavity(mesh, uv, vv, u_lid, v_lid)
    return uv, vv, pv


def cal_momemtum_link_coeff(mesh, ap, scx, scy, mdotf, ubc, vbc):
    global mu
    area, delta, snsign = uf.shorted_name(mesh.global_faces, 'area', 'delta', 'snsign')

    mask_bc, mask_in = mesh.get_face_mask_element_wise()
    (_area, _delta, _ubc, _vbc, mf) = mesh.var_elem_wise(area, delta, ubc, vbc, mdotf)
    ap.fill(0.0), scx.fill(0.0), scy.fill(0.0)

    mf *= snsign
    ap_in = (mu * _area / _delta + 0.5 * (np.abs(mf) + mf)) * mask_in
    ap_bc = (mu * _area / _delta) * mask_bc
    ap = ap_in + ap_bc
    anb = (- mu * _area / _delta - 0.5 * (np.abs(mf) - mf)) * mask_in
    scx = (_ubc * mu * _area / _delta - mf * _ubc) * mask_bc
    scy = (_vbc * mu * _area / _delta - mf * _vbc) * mask_bc

    (ap, scx, scy) = uf.get_total_flux(ap, scx, scy)
    return ap, anb, scx, scy


def cal_momentume_pressure_source(mesh, fw, scx, scy, pc, pf):
    area, sn, snsign = uf.shorted_name(mesh.global_faces, 'area', 'sn', 'snsign')

    # Calculate pressure face value
    pf = cal_face_value(mesh, fw, pc, pf)
    (_pf, _area, _sn) = mesh.var_elem_wise(pf, area, sn)

    scx -= uf.get_total_flux(_pf * _sn[:,:,0] * snsign * _area, )[0]
    scy -= uf.get_total_flux(_pf * _sn[:,:,1] * snsign * _area, )[0]

    return scx, scy


def cal_momentum_skew_term(uv, vv, mesh):
    centroid = uf.shorted_name(mesh.elems, 'centroid')[0]
    st, snsign, delta = uf.shorted_name(mesh.global_faces, 'st', 'snsign', 'delta')
    c2f, f2c, f2v, f_2_bf = uf.shorted_name(mesh.link, 'c2f', 'f2c', 'f2v', 'f_2_bf')

    mask_bc, mask_in = mesh.get_face_mask_element_wise()
    (_delta, _st, _uv, _vv, _f2c, _f2v) = mesh.var_elem_wise(delta, st, uv, vv, f2c, f2v)
    d1 = centroid[_f2c][:,:,1,:] - centroid[_f2c][:,:,0,:]
    tdotl = np.sum(_st * d1, axis = 2)

    sumfx = (tdotl * (uv[_f2v][:, :, 1] - uv[_f2v][:, :, 0]) * snsign / _delta) * mask_in
    sumfy = (tdotl * (vv[_f2v][:, :, 1] - vv[_f2v][:, :, 0]) * snsign / _delta) * mask_in

    skewx = uf.get_total_flux(sumfx)[0] * mu
    skewy = uf.get_total_flux(sumfy)[0] * mu

    return skewx, skewy


def solve_mom_eq(mesh, varc_, ap_, anb_, sc_, skew_, res_, tag):
    snsign = uf.shorted_name(mesh.global_faces, 'snsign')[0]
    c2f, f2c = uf.shorted_name(mesh.link, 'c2f', 'f2c')
    def cal_residual(mesh, varc, ap, anb, sc, skew, res, res2_init, iter_):
        (_f2c,) = mesh.var_elem_wise(f2c, )
        icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])   # Neighbor cells
        sumf = uf.get_total_flux(anb * varc[icn])[0]
        res = sc + skew - ap * varc - sumf

        res2 = np.linalg.norm(res)

        if iter_ == 0:
            res2_init = res2

        return res, res2, res2_init

    def cal_jacobi_loop(varc, ap, res, rin_uv):
        var_tilda = res / (ap * (1.0 + rin_uv))

        assert not np.any(np.isnan(varc + var_tilda)), f"{tag} momentum - NAN value"
        assert not np.any(np.isinf(varc + var_tilda)), f"{tag} momentum - INF value"
        return varc + var_tilda

    res2_init = 0.0
    for iter_ in range(iter_mom):
        res_, res2, res2_init = cal_residual(mesh, varc_, ap_, anb_, sc_, skew_, res_, res2_init, iter_)

        varc_ = cal_jacobi_loop(varc_, ap_, res_, rin_uv = 0.1)

        if res2_init == 0.0:
            print(f"\t{tag}-mom converged at {iter_} iteration")
            break
        if res2/res2_init < tol_inner:
            print(f"\t{tag}-mom converged at {iter_} iteration")
            break

    return varc_


def cal_massflow_face(mesh, uc, vc, pc, pf, ap, fw, mdotf):
    """
    Calculate face mass flow using PWIM method.
    Ref: https://www.youtube.com/watch?v=4jQxtz29UQw&list=PLVuuXJfoPgT4gJcBAAFPW7uMwjFKB9aqT&t=1243s

    Symbol explanation:
    - In order to vectorize formulations, I used the combination symbols to mark array shape, it this function,
    they include three types:
        + _var  implies this array's head is shaped (number of elements) x (number of local faces).
        +  var_ implies this array's head is shaped (number of global faces) x (2).
        + _var_ implies this array's head is shaped (number of elements) x (2) x (number of local faces).
    The tail (number of elements along the latest axis) of the var array depends on its original tail, for example:
        + sn has tail == 2, so _sn = (number of elements) x (number of local faces) x (2)
        + area has tail == 1, so _area = (number of elements) x (number of local faces) x (1)
    """
    def grad_2nd(var, delta__):
        assert len(var) == len(delta), "TypeError: Segmentation fault"
        return (var[..., 1] - var[..., 0]) / delta__

    sn, snsign, area, delta = uf.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area', 'delta')
    c2f, f2c, f_2_bf = uf.shorted_name(mesh.link, 'c2f', 'f2c', 'f_2_bf')
    volume = uf.shorted_name(mesh.elems, 'volume')[0]

    # Weighted cell components
    (uc_, vc_) = mesh.var_face_wise(uc, vc)
    weighted_coeff = np.array([fw, 1 - fw]).T
    velf_x = np.sum(weighted_coeff * uc_, axis=1)
    velf_y = np.sum(weighted_coeff * vc_, axis=1)
    velf_ic = np.einsum('ij,ij->i', np.array([velf_x,velf_y]).T, sn)

    # Weighted cell pressure components
    (_pf, _area, _sn) = mesh.var_elem_wise(pf, area, sn)
    (_pf_, _area_, _sn_, snsign_, ap_) = mesh.var_face_wise(_pf, _area, _sn, snsign, ap)
    Vdp_x = np.sum(_pf_ * _area_ * _sn_[:,:,:, 0] * snsign_, axis=2)
    Vdp_y = np.sum(_pf_ * _area_ * _sn_[:,:,:, 1] * snsign_, axis=2)
    velf_x = np.sum(weighted_coeff * Vdp_x / ap_, axis=1)
    velf_y = np.sum(weighted_coeff * Vdp_y / ap_, axis=1)
    velf_pc = np.einsum('ij,ij->i', np.array([velf_x,velf_y]).T, sn)

    # Weighted face pressure components
    (ap_, vol_, pc_) = mesh.var_face_wise(ap, volume, pc)
    velf_pf =   np.sum(weighted_coeff * vol_ / ap_, axis=1) * grad_2nd(pc_, delta)

    maskf_in = f_2_bf < 0
    vdotn = velf_ic + velf_pc - velf_pf
    mdotf = vdotn * rho * area * maskf_in

    return mdotf


def cal_source_mass_imbalance(mesh, sc_p, mdotf):    # Calculate source due to mass imbalance
    snsign = uf.shorted_name(mesh.global_faces, 'snsign')[0]
    c2f = uf.shorted_name(mesh.link, 'c2f')[0]
    sc_p.fill(0.0)

    (_mdotf, ) = mesh.var_elem_wise(mdotf)
    sc_p -= np.sum(_mdotf * snsign, axis=1)

    return sc_p

def cal_pressure_link_coeff(mesh, ap, ap_p, anb_p, fw):
    snsign, area, delta = uf.shorted_name(mesh.global_faces, 'snsign', 'area', 'delta')
    c2f, f2c, f_2_bf = uf.shorted_name(mesh.link, 'c2f', 'f2c', 'f_2_bf')
    volume = uf.shorted_name(mesh.elems, 'volume')[0]
    ap_p.fill(0.0), anb_p.fill(0.0)

    mask_bc, mask_in = mesh.get_face_mask_element_wise()
    (_f2c, _area, _delta, _fw) = mesh.var_elem_wise(f2c, area, delta, fw)
    ic = np.tile(np.arange(mesh.no_elems()), (mesh.no_local_faces(), 1)).T
    icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])
    ap_p  = (  (_fw * volume[ic] / ap[ic] + (1.0 - _fw) * volume[icn] / ap[icn]) * rho * _area / _delta) * mask_in
    anb_p = (- (_fw * volume[ic] / ap[ic] + (1.0 - _fw) * volume[icn] / ap[icn]) * rho * _area / _delta) * mask_in

    return np.sum(ap_p, axis=1), anb_p

def cal_pressure_link_coeff1(mesh, ap, ap_p, anb_p, fw):
    snsign, area, delta = uf.shorted_name(mesh.global_faces, 'snsign', 'area', 'delta')
    c2f, f2c, f_2_bf = uf.shorted_name(mesh.link, 'c2f', 'f2c', 'f_2_bf')
    volume = uf.shorted_name(mesh.elems, 'volume')[0]
    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = c2f[ic, local_face]
            if f_2_bf[ifc] != -1:  # IF boundary face = continue
                continue
            if snsign[ic, local_face] == 1:
                icn = f2c[ifc, 1]
            else:
                icn = f2c[ifc, 0]
            ap_p[ic] += (fw[ifc] * volume[ic] / ap[ic] + (1.0 - fw[ifc]) * volume[icn] / ap[icn]) \
                            * rho * area[ifc] / delta[ifc]
            anb_p[ic, local_face] = - (fw[ifc] * volume[ic] / ap[ic] + (1.0 - fw[ifc]) * volume[icn] / ap[icn]) \
                            * rho * area[ifc] / delta[ifc]

    return ap_p, anb_p

def solve_poison_eq(mesh, pcor_, ap_, anb_p_, ap_p_, sc_p_, res_p_, mdotf_, fw):
    snsign = uf.shorted_name(mesh.global_faces, 'snsign')[0]
    c2f, f2c = uf.shorted_name(mesh.link, 'c2f', 'f2c')

    def cal_jacobi_loop(mesh, pcor, anb_p, ap_p, sc_p):
        (_f2c,) = mesh.var_elem_wise(f2c, )
        icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])  # Neighbor cells
        sumf = uf.get_total_flux(anb_p * pcor[icn])[0]
        pcor = relax_p * (sc_p - sumf) / ap_p

        return pcor

    def cal_residual(pcor, pcor_old, res2p, iter_):
        res = np.linalg.norm(pcor - pcor_old)
        res2 = np.sqrt(np.maximum(0.0, res))
        if iter_ == 0:
            res2p = res2

        return res2, res2p

    pcor_.fill(0.0)
    sc_p_ = cal_source_mass_imbalance(mesh, sc_p_, mdotf_)
    ap_p_, anb_p_ = cal_pressure_link_coeff(mesh, ap_, ap_p_, anb_p_, fw)

    res2p = 0.0
    for iter_ in range(iter_pp):
        pcor_old = pcor_.copy()
        pcor_ = cal_jacobi_loop(mesh, pcor_, anb_p_, ap_p_, sc_p_)
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
    sn, snsign, area = uf.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area')

    # Calculate correction pressure face value
    pfcor = cal_face_value(mesh, fw, pcor, pfcor)

    (_pfcor, _area, _sn) = mesh.var_elem_wise(pfcor, area, sn)

    ucor.fill(0.0), vcor.fill(0.0)
    ucor = uf.get_total_flux(_pfcor * _sn[:, :, 0] * snsign * _area, )[0]
    vcor = uf.get_total_flux(_pfcor * _sn[:, :, 1] * snsign * _area, )[0]

    ucor = - ucor / ap
    vcor = - vcor / ap
    uc += relax_uv * ucor
    vc += relax_uv * vcor

    return uc, vc

def corrected_cell_vel1(mesh, ucor, vcor, pcor, uc, vc, pfcor, ap, fw):
    sn, snsign, area = uf.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area')
    c2f, f2c = uf.shorted_name(mesh.link, 'c2f', 'f2c')

    # Calculate correction pressure face value
    pfcor = cal_face_value(mesh, fw, pcor, pfcor)
    for ic in range(mesh.no_elems()):
        for local_face in range(mesh.no_local_faces()):
            ifc = c2f[ic, local_face]
            ucor[ic] += pfcor[ifc] * sn[ifc, 0] * snsign[ic, local_face] * area[ifc]
            vcor[ic] += pfcor[ifc] * sn[ifc, 1] * snsign[ic, local_face] * area[ifc]

        ucor[ic] = - ucor[ic] / ap[ic]
        vcor[ic] = - vcor[ic] / ap[ic]
        uc[ic] += relax_uv * ucor[ic]
        vc[ic] += relax_uv * vcor[ic]

    return uc, vc


def corrected_massflux(mesh, ap, fw, mdotfcor, pcor, mdotf):
    area, delta = uf.shorted_name(mesh.global_faces, 'area', 'delta')
    f2c, f_2_bf = uf.shorted_name(mesh.link, 'f2c', 'f_2_bf')
    volume = uf.shorted_name(mesh.elems, 'volume')[0]
    weighted_coeff = np.array([fw, 1 - fw]).T
    maskf_in = f_2_bf < 0
    (vol_, ap_, pcor_) = mesh.var_face_wise(volume, ap, pcor)
    coeff = np.einsum('ij,ij->i', weighted_coeff, vol_ * ap_)
    mdotfcor = (rho * coeff * area * (pcor_[:, 0] - pcor_[:, 1]) / delta) * maskf_in
    mdotf += relax_uv * mdotfcor

    return mdotf

def corrected_massflux1(mesh, ap, fw, mdotfcor, pcor, mdotf):
    area, delta = uf.shorted_name(mesh.global_faces, 'area', 'delta')
    f2c, f_2_bf = uf.shorted_name(mesh.link, 'f2c', 'f_2_bf')
    volume = uf.shorted_name(mesh.elems, 'volume')[0]
    for ifc in range(mesh.no_faces()):
        if f_2_bf[ifc] != -1:  # IF boundary face = continue
            continue
        c0 = f2c[ifc, 0]
        c1 = f2c[ifc, 1]
        coeff = fw[ifc] * volume[c0] / ap[c0] + (1.0 - fw[ifc]) * volume[c1] / ap[c1]
        mdotfcor[ifc] = rho * coeff * area[ifc] * (pcor[c0] - pcor[c1]) / delta[ifc]
        mdotf[ifc] += relax_uv * mdotfcor[ifc]

    return mdotf


def corrected_pressure(pc, pf, pcor, pfcor):
    pc = pc + relax_p * pcor
    # pf = pf + relax_p * pfcor
    return pc, pf


def cal_outer_res(uc, vc, uc_old, vc_old):
    error_u = np.sum(np.abs(uc - uc_old)) / np.sum(np.abs(uc))
    if np.sum(np.abs(vc)) < 1e-09:
        error_v = 0.0
    else:
        error_v = np.sum(np.abs(vc - vc_old)) / np.sum(np.abs(vc))

    return error_u, error_v


