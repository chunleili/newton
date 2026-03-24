"""Warp kernels for volumetric muscle (vmuscle) in VBD solver.

Implements DeGroote-Fregly 2016 (DGF) Hill-type muscle model as anisotropic
fiber energy on top of the standard Neo-Hookean elasticity.

References:
    - DeGroote et al. (2016), "Evaluation of Direct Collocation Optimal Control
      Problem Formulations for Solving the Muscle Redundancy Problem"
    - Smith et al., "Stable Neo-Hookean Flesh Simulation"
    - Kim et al., "Anisotropic Elasticity for Inversion-Safety and Element
      Rehabilitation"
"""

import warp as wp

from .particle_vbd_kernels import (
    ParticleForceElementAdjacencyInfo,
    assemble_tet_vertex_force_and_hessian,
    get_vertex_adjacent_tet_id_order,
    get_vertex_num_adjacent_tets,
    mat99,
    vec9,
)


# =============================================================================
# DGF Curve Functions
# =============================================================================


@wp.func
def dgf_active_force_length(lm_tilde: float) -> float:
    """DGF 2016 active force-length curve (sum of 3 Gaussians)."""
    EPS = 1.0e-6
    # Gaussian 1: b1=0.815, b2=1.055, b3=0.162, b4=0.063
    denom1 = wp.max(wp.abs(0.162 + 0.063 * lm_tilde), EPS)
    t1 = (lm_tilde - 1.055) / denom1
    g1 = 0.815 * wp.exp(-0.5 * t1 * t1)
    # Gaussian 2: b1=0.433, b2=0.717, b3=-0.030, b4=0.200
    denom2 = wp.max(wp.abs(-0.030 + 0.200 * lm_tilde), EPS)
    t2 = (lm_tilde - 0.717) / denom2
    g2 = 0.433 * wp.exp(-0.5 * t2 * t2)
    # Gaussian 3: b1=0.1, b2=1.0, b3=0.354, b4=0.0
    t3 = (lm_tilde - 1.0) / 0.354
    g3 = 0.1 * wp.exp(-0.5 * t3 * t3)
    return g1 + g2 + g3


@wp.func
def dgf_active_force_length_deriv(lm_tilde: float) -> float:
    """Derivative of f_L w.r.t. l_tilde.

    d/dl [b1*exp(-0.5*t^2)] where t = (l-b2)/(b3+b4*l)
    dt/dl = (b3 + b4*b2) / (b3+b4*l)^2
    """
    EPS = 1.0e-6
    result = float(0.0)
    # Gaussian 1
    d1 = wp.max(wp.abs(0.162 + 0.063 * lm_tilde), EPS)
    t1 = (lm_tilde - 1.055) / d1
    dtdl1 = (0.162 + 0.063 * 1.055) / (d1 * d1)
    result += 0.815 * wp.exp(-0.5 * t1 * t1) * (-t1) * dtdl1
    # Gaussian 2
    d2 = wp.max(wp.abs(-0.030 + 0.200 * lm_tilde), EPS)
    t2 = (lm_tilde - 0.717) / d2
    dtdl2 = (-0.030 + 0.200 * 0.717) / (d2 * d2)
    result += 0.433 * wp.exp(-0.5 * t2 * t2) * (-t2) * dtdl2
    # Gaussian 3
    d3 = 0.354
    t3 = (lm_tilde - 1.0) / d3
    dtdl3 = 1.0 / d3
    result += 0.1 * wp.exp(-0.5 * t3 * t3) * (-t3) * dtdl3
    return result


@wp.func
def dgf_passive_force(lm_tilde: float) -> float:
    """DGF 2016 passive force-length curve (exponential)."""
    kPE = 4.0
    e0 = 0.6
    lm_min = 0.2
    offset = wp.exp(kPE * (lm_min - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    result = (wp.exp(kPE * (lm_tilde - 1.0) / e0) - offset) / denom
    return wp.max(result, 0.0)


@wp.func
def dgf_passive_force_deriv(lm_tilde: float) -> float:
    """Derivative of f_PE w.r.t. l_tilde."""
    kPE = 4.0
    e0 = 0.6
    lm_min = 0.2
    offset = wp.exp(kPE * (lm_min - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    return (kPE / e0) * wp.exp(kPE * (lm_tilde - 1.0) / e0) / denom


@wp.func
def dgf_force_velocity(v_norm: float) -> float:
    """DGF 2016 force-velocity curve (sinh^-1 form)."""
    d1 = -0.3211346127989808
    d2 = -8.149
    d3 = -0.374
    d4 = 0.8825327733249912
    x = d2 * v_norm + d3
    return d1 * wp.log(x + wp.sqrt(x * x + 1.0)) + d4


# =============================================================================
# Energy Derivatives
# =============================================================================


@wp.func
def fiber_energy_derivatives(
    I5: float, activation: float, sigma0: float, fv_factor: float
) -> wp.vec2:
    """Compute Psi'(I5) and Psi''(I5) for Hill-type fiber energy.

    Returns:
        vec2: (dPsi_dI5, d2Psi_dI5_2)
    """
    lam = wp.sqrt(wp.max(I5, 1.0e-8))

    fL = dgf_active_force_length(lam)
    fPE = dgf_passive_force(lam)
    dfL = dgf_active_force_length_deriv(lam)
    dfPE = dgf_passive_force_deriv(lam)

    # Psi'(I5) = sigma0/(2*lam) * [a*f_L*f_V + f_PE]
    total_force = activation * fL * fv_factor + fPE
    dPsi = sigma0 / (2.0 * lam) * total_force
    dPsi = wp.max(dPsi, 0.0)  # clamp to ensure PSD Hessian

    # Psi''(I5) = sigma0/(4*lam^3) * [a*f_V*(f_L'*lam - f_L) + (f_PE'*lam - f_PE)]
    d2_term = activation * fv_factor * (dfL * lam - fL) + (dfPE * lam - fPE)
    d2Psi = sigma0 / (4.0 * lam * lam * lam) * d2_term
    # Clamp d2Psi >= 0 to ensure PSD Hessian (SPD projection).
    # Negative d2Psi from active force derivative can make Hessian indefinite.
    d2Psi = wp.max(d2Psi, 0.0)

    return wp.vec2(dPsi, d2Psi)


# =============================================================================
# PK1 Stress and Hessian
# =============================================================================


@wp.func
def evaluate_fiber_pk1_and_hessian(
    F: wp.mat33,
    fiber_dir: wp.vec3,
    activation: float,
    sigma0: float,
    rest_volume: float,
    fv_factor: float,
) -> tuple[vec9, mat99]:
    """Compute fiber PK1 stress (vec9) and 9x9 Hessian, scaled by rest_volume.

    Fiber energy: W = Psi(I5) where I5 = |F d|^2
    PK1: P = 2*Psi'*F*A, A = d (x) d
    Hessian: H = 2*Psi'*kron(A, I3) + 4*Psi''*outer(vec(FA), vec(FA))
    """
    d0 = fiber_dir
    A = wp.outer(d0, d0)

    Fd0 = F * d0
    I5 = wp.dot(Fd0, Fd0)

    derivs = fiber_energy_derivatives(I5, activation, sigma0, fv_factor)
    dPsi = derivs[0]
    d2Psi = derivs[1]

    # PK1 stress: P = 2*Psi'*F*A
    FA = F * A
    P = FA * (2.0 * dPsi)
    P_vec = vec9(
        P[0, 0], P[1, 0], P[2, 0],
        P[0, 1], P[1, 1], P[2, 1],
        P[0, 2], P[1, 2], P[2, 2],
    )
    P_vec = P_vec * rest_volume

    # 9x9 Hessian
    FA_vec = vec9(
        FA[0, 0], FA[1, 0], FA[2, 0],
        FA[0, 1], FA[1, 1], FA[2, 1],
        FA[0, 2], FA[1, 2], FA[2, 2],
    )

    # Term 1: 2*Psi'*kron(A, I3)
    H = mat99()
    for ci in range(3):
        for cj in range(3):
            a_val = A[ci, cj] * 2.0 * dPsi
            for kk in range(3):
                H[ci * 3 + kk, cj * 3 + kk] = a_val

    # Term 2: 4*Psi''*outer(vec(FA), vec(FA))
    for i in range(9):
        for j in range(9):
            H[i, j] = H[i, j] + 4.0 * d2Psi * FA_vec[i] * FA_vec[j]

    H = H * rest_volume
    return P_vec, H


# =============================================================================
# Per-vertex Force and Hessian from Fiber Energy
# =============================================================================


@wp.func
def evaluate_fiber_force_and_hessian(
    tet_id: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    Dm_inv: wp.mat33,
    fiber_dir: wp.vec3,
    sigma0: float,
    activation: float,
    fv_factor: float,
    fiber_damping: float,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Compute per-vertex (vec3 force, mat33 hessian) for one adjacent tet."""
    i0 = tet_indices[tet_id, 0]
    i1 = tet_indices[tet_id, 1]
    i2 = tet_indices[tet_id, 2]
    i3 = tet_indices[tet_id, 3]

    # Get vertices
    v0 = pos[i0]
    v1 = pos[i1]
    v2 = pos[i2]
    v3 = pos[i3]

    # Compute rest volume from Dm_inv
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    # Deformation gradient
    Ds = wp.matrix_from_cols(v1 - v0, v2 - v0, v3 - v0)
    F = Ds * Dm_inv

    # Fiber PK1 and Hessian
    P_vec, H = evaluate_fiber_pk1_and_hessian(
        F, fiber_dir, activation, sigma0, rest_volume, fv_factor
    )

    # Extract barycentric weight for this vertex (same pattern as Neo-Hookean)
    if v_order == 0:
        m = wp.vec3(
            -(Dm_inv[0, 0] + Dm_inv[1, 0] + Dm_inv[2, 0]),
            -(Dm_inv[0, 1] + Dm_inv[1, 1] + Dm_inv[2, 1]),
            -(Dm_inv[0, 2] + Dm_inv[1, 2] + Dm_inv[2, 2]),
        )
    elif v_order == 1:
        m = wp.vec3(Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2])
    elif v_order == 2:
        m = wp.vec3(Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2])
    else:
        m = wp.vec3(Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2])

    force, hessian = assemble_tet_vertex_force_and_hessian(
        P_vec, H, m[0], m[1], m[2]
    )

    # Fiber damping: viscous force along fiber direction proportional to stretch rate.
    # Derivation: let Fd0 = F * d0, l = |Fd0|. Then d(Fd0)/dx = dot(m, d0) * I
    # (since m encodes the barycentric weight for this vertex in Dm_inv),
    # so dl/dx = dot(m, d0) * normalize(Fd0).
    if fiber_damping > 0.0:
        Fd0 = F * fiber_dir
        l_cur = wp.sqrt(wp.max(wp.dot(Fd0, Fd0), 1.0e-8))

        Ds_prev = wp.matrix_from_cols(
            pos_prev[i1] - pos_prev[i0],
            pos_prev[i2] - pos_prev[i0],
            pos_prev[i3] - pos_prev[i0],
        )
        l_prev = wp.sqrt(wp.max(wp.dot((Ds_prev * Dm_inv) * fiber_dir,
                                       (Ds_prev * Dm_inv) * fiber_dir), 1.0e-8))

        dl_dt = (l_cur - l_prev) / dt
        dldx = wp.dot(m, fiber_dir) * (Fd0 / l_cur)

        kd = fiber_damping * sigma0 * rest_volume
        force = force + (-kd * dl_dt) * dldx
        hessian = hessian + (kd / dt) * wp.outer(dldx, dldx)

    return force, hessian


# =============================================================================
# Inline F-V Factor Computation
# =============================================================================


@wp.func
def compute_fv_factor_inline(
    tet_id: int,
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    Dm_inv: wp.mat33,
    fiber_dir: wp.vec3,
    q_prev: wp.array(dtype=wp.vec3),
    q_prev2: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
) -> float:
    """Compute f_V factor from two consecutive timestep positions (inline)."""
    i0 = tet_indices[tet_id, 0]
    i1 = tet_indices[tet_id, 1]
    i2 = tet_indices[tet_id, 2]
    i3 = tet_indices[tet_id, 3]

    # l_tilde_current from prev positions x(n-1)
    Ds1 = wp.matrix_from_cols(
        q_prev[i1] - q_prev[i0], q_prev[i2] - q_prev[i0], q_prev[i3] - q_prev[i0]
    )
    Fd1 = (Ds1 * Dm_inv) * fiber_dir
    l_current = wp.sqrt(wp.max(wp.dot(Fd1, Fd1), 1.0e-8))

    # l_tilde_prev from prev2 positions x(n-2)
    Ds2 = wp.matrix_from_cols(
        q_prev2[i1] - q_prev2[i0],
        q_prev2[i2] - q_prev2[i0],
        q_prev2[i3] - q_prev2[i0],
    )
    Fd2 = (Ds2 * Dm_inv) * fiber_dir
    l_prev = wp.sqrt(wp.max(wp.dot(Fd2, Fd2), 1.0e-8))

    v_norm = wp.clamp((l_current - l_prev) / (dt * v_max), -1.0, 1.0)
    return dgf_force_velocity(v_norm)


# =============================================================================
# Main Accumulation Kernel
# =============================================================================


@wp.kernel
def accumulate_fiber_force_and_hessian(
    dt: float,
    v_max: float,
    fiber_damping: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q_prev2: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    vmuscle_tet_fiber_dirs: wp.array(dtype=wp.vec3),
    vmuscle_tet_sigma0: wp.array(dtype=wp.float32),
    vmuscle_tet_activations: wp.array(dtype=wp.float32),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
    # output (accumulated)
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    """Accumulate Hill-type fiber force and Hessian for each particle in color group.

    F-V factor is computed inline from particle_q_prev and particle_q_prev2,
    no precomputation needed.
    """
    tid = wp.tid()
    particle_index = particle_ids_in_color[tid]
    f = wp.vec3(0.0, 0.0, 0.0)
    h = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
    for adj in range(num_adj_tets):
        tet_id, v_order = get_vertex_adjacent_tet_id_order(
            particle_adjacency, particle_index, adj
        )
        sigma0 = vmuscle_tet_sigma0[tet_id]
        if sigma0 > 0.0:
            fiber_dir = vmuscle_tet_fiber_dirs[tet_id]
            Dm_inv = tet_poses[tet_id]
            activation = vmuscle_tet_activations[tet_id]

            # Inline F-V: compute from prev and prev2 positions
            fv_factor = compute_fv_factor_inline(
                tet_id, tet_indices, Dm_inv, fiber_dir,
                particle_q_prev, particle_q_prev2, dt, v_max,
            )

            f_fiber, h_fiber = evaluate_fiber_force_and_hessian(
                tet_id, v_order, pos, particle_q_prev, tet_indices, Dm_inv,
                fiber_dir, sigma0, activation, fv_factor,
                fiber_damping, dt,
            )
            f = f + f_fiber
            h = h + h_fiber

    particle_forces[particle_index] = particle_forces[particle_index] + f
    particle_hessians[particle_index] = particle_hessians[particle_index] + h
