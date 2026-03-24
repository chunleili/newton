"""Kernel launch wrappers for volumetric muscle (vmuscle) in VBD solver."""

import warp as wp

from .vmuscle_kernels import accumulate_fiber_force_and_hessian


def launch_accumulate_fiber_force_and_hessian(
    model,
    tet_activations,
    max_contraction_velocity,
    fiber_damping,
    dt,
    color_group,
    particle_q_prev,
    particle_q_prev2,
    pos,
    particle_adjacency,
    particle_forces,
    particle_hessians,
    device,
):
    """Accumulate fiber force and Hessian for one color group.

    Args:
        model: Newton Model with vmuscle properties.
        tet_activations: Per-tet activation array from Control.tet_activations.
        max_contraction_velocity: V_max scalar [l_opt/s].
        fiber_damping: Fiber viscous damping coefficient.
        dt: Time step size.
    """
    vmuscle = model.vmuscle
    wp.launch(
        kernel=accumulate_fiber_force_and_hessian,
        dim=color_group.size,
        inputs=[
            dt,
            max_contraction_velocity,
            fiber_damping,
            color_group,
            particle_q_prev,
            particle_q_prev2,
            pos,
            model.tet_indices,
            model.tet_poses,
            vmuscle.fiber_dirs,
            vmuscle.sigma0,
            tet_activations,
            particle_adjacency,
        ],
        outputs=[particle_forces, particle_hessians],
        device=device,
    )
