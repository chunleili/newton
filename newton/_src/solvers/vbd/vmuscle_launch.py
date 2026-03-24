"""Kernel launch wrappers for volumetric muscle (vmuscle) in VBD solver."""

import warp as wp

from .vmuscle_kernels import accumulate_fiber_force_and_hessian


def launch_accumulate_fiber_force_and_hessian(
    model,
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
    """Accumulate fiber force and Hessian for one color group."""
    fiber_damping = getattr(model, 'vmuscle_fiber_damping', 0.0)
    wp.launch(
        kernel=accumulate_fiber_force_and_hessian,
        dim=color_group.size,
        inputs=[
            dt,
            model.vmuscle_max_contraction_velocity,
            fiber_damping,
            color_group,
            particle_q_prev,
            particle_q_prev2,
            pos,
            model.tet_indices,
            model.tet_poses,
            model.vmuscle_tet_fiber_dirs,
            model.vmuscle_tet_sigma0,
            model.vmuscle_tet_activations,
            particle_adjacency,
        ],
        outputs=[particle_forces, particle_hessians],
        device=device,
    )
