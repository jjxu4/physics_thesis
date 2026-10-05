import numpy as np


class LC_series_in_para:

    def __init__(self, capacitance, voltage_0, loop_system, num_left_branch, num_right_branch):
        self.capacitance    = capacitance
        self.v0      = voltage_0
        self.loop_system    = loop_system
        self.num_left_branch    = num_left_branch
        self.num_right_branch   = num_right_branch

    def anim_loop_voltage(self, radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res):

        M = self.loop_system.calc_inductance_mtrx()
        N_left = self.num_left_branch
        N = N_left + self.num_right_branch

        # Each entry sums the flux linkages for a pair of series branches.
        M_branches = np.array([
            [np.sum(M[:N_left, :N_left]), np.sum(M[:N_left, N_left:N])],
            [np.sum(M[N_left:N, :N_left]), np.sum(M[N_left:N, N_left:N])]
        ])
        M_inv = np.linalg.inv(M_branches)
        w = np.sqrt((np.ones((1, 2)) @ (M_inv @ np.ones((2, 1)))) / self.capacitance)

        def I_branches(t):
            return (-1 * self.v0 / w) * np.sin(w * t) * (M_inv @ np.ones((2, 1)))

        def dIdt(t):
            return (-1 * self.v0) * np.cos(w * t) * (M_inv @ np.ones((2, 1)))

        for i in range(N):
            branch = 0 if i < N_left else 1
            self.loop_system.loops[i].update_fields(lambda t, branch=branch: I_branches(t)[branch], lambda t, branch=branch: dIdt(t)[branch])


        return self.loop_system.animate_loop_voltage(radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res)
