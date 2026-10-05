import numpy as np


class LC_para:

    def __init__(self, capacitance, voltage_0, loop_system):
        self.capacitance    = capacitance
        self.v0      = voltage_0
        self.loop_system    = loop_system

    def anim_loop_voltage(self, radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res):

        M = self.loop_system.calc_inductance_mtrx()
        M_inv = np.linalg.inv(M)
        N = len(self.loop_system.loops)                       # number of loops
        w = np.sqrt((np.ones((1, N)) @ (M_inv @ np.ones((N,1)) ))/self.capacitance)

        def I_loops(t):
            return (-1 * self.v0 / w) * np.sin(w * t) * (M_inv @ np.ones((N, 1)))

        def dIdt(t):
            return (-1 * self.v0 ) * np.cos(w * t) * (M_inv @ np.ones((N, 1)))

        for i in range(N):
            self.loop_system.loops[i].update_fields(lambda t, i=i: I_loops(t)[i], lambda t, i=i: dIdt(t)[i])


        return self.loop_system.animate_loop_voltage(radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res)