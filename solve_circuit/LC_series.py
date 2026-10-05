import numpy as np


class LC_series:

    def __init__(self, capacitance, voltage_0, loop_system):
        self.capacitance    = capacitance
        self.v0      = voltage_0
        self.loop_system    = loop_system

    def anim_loop_voltage(self, radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res):

        M = self.loop_system.calc_inductance_mtrx()
        N = len(self.loop_system.loops)                       # number of loops
        L = np.sum(M)                                       # includes all mutual inductances
        w = 1 / np.sqrt(self.capacitance * L)

        def current(t):
            return (-1 * self.capacitance * self.v0 * w) * np.sin(w * t)

        def dIdt(t):
            return (-1 * self.capacitance * self.v0 * w**2) * np.cos(w * t)

        for loop in self.loop_system.loops:
            loop.update_fields(current, dIdt)


        return self.loop_system.animate_loop_voltage(radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res)
