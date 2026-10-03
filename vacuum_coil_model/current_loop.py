from scipy.special import ellipk, ellipe
from scipy.constants import pi, mu_0
import numpy as np

class current_loop:

    def __init__(self, name, radius, z_pos, current, dIdt, wire_radius):
        '''
        name:           string for the name
        radius:         float for the radius
        z_pos:          float for the z_pos
        current:        callable function I(t)
        dIdt:           callable function dI/dt (t)
        wire_radius:    float for wire thickness
        '''
        self.name           = name
        self.radius         = radius
        self.z_pos          = z_pos
        self.wire_radius    = wire_radius

        self.a_phi          = None
        self.b_r            = None
        self.b_z            = None
        self._create_a_phi()
        self._create_b_r()
        self._create_b_z()


        self.Bfield_R       = None
        self.Bfield_Z       = None
        self.Efield_phi     = None
        self._create_fields(current, dIdt)

    def _create_fields(self, current, dIdt):

        def E_field_phi(R, Z, t):
            '''
            R and Z expected to be meshgrids created by np.meshgrid(R, Z).
            Returns grid in same shape as R and Z with values of E_phi filled in
            '''

            return -1 * dIdt(t)[:, None, None] * self.a_phi(R, Z)

        self.Efield_phi = E_field_phi

        def B_field_R(R, Z, t):
            '''
            R and Z expected to be meshgrids created by np.meshgrid(R, Z).
            Returns grid in same shape as R and Z with values of Bfield_R filled in
            '''
            return current(t)[:, None, None] * self.b_r(R, Z)

        self.Bfield_R = B_field_R

        def B_field_Z(R, Z, t):
            '''
            R and Z expected to be meshgrids created by np.meshgrid(R, Z).
            Returns grid in same shape as R and Z with values of Bfield_Z filled in
            '''
            return current(t)[:, None, None] * self.b_z(R, Z)

        self.Bfield_Z = B_field_Z

    # ================== Helpers ==================
    def _create_a_phi(self):

        def a_phi(R, Z):
            '''
            R and Z expected to be meshgrids created by np.meshgrid(R, Z).
            Returns grid in same shape as R and Z with values of a_phi filled in
            '''
            D = (self.radius + R)**2 + Z**2
            m = (4 * self.radius * R)/D

            near_wire = (R - self.radius)**2 + (Z)**2 < self.wire_radius**2
            near_axis = np.abs(R) <= (self.radius * 0.01) & ~near_wire
            mask = near_axis | near_wire

            a_phi_val = np.full_like(R, np.nan, dtype=float)

            a_phi_val[~mask] = ((mu_0 * np.sqrt(D[~mask]))/(2 * pi * R[~mask])) * ((1 - 0.5 * m[~mask]) * ellipk(m[~mask]) - ellipe(m[~mask]))
            a_phi_val[near_axis] = (mu_0 * self.radius**2 * R[near_axis])/(4 * (self.radius**2 + Z[near_axis]**2)**(3/2))

            return a_phi_val

        def a_phi_shifted(R,Z):
            return a_phi(R, Z - self.z_pos)
        
        self.a_phi = a_phi_shifted

    def _create_b_r(self):

        def b_r(R,Z):
            '''
            R and Z expected to be meshgrids created by np.meshgrid(R, Z).
            Returns grid in same shape as R and Z with values of b_r filled in
            '''
            D = (self.radius + R)**2 + Z**2
            m = (4 * self.radius * R)/D
            q = (self.radius - R)**2 + Z**2

            near_wire = (R - self.radius)**2 + (Z)**2 < self.wire_radius**2
            near_axis = np.abs(R) < 1e-3
            mask = near_axis | near_wire

            b_r_val = np.full_like(R, np.nan, dtype=float)

            b_r_val[~mask] = ((mu_0 * Z[~mask])/(2 * pi * R[~mask] * np.sqrt(D[~mask]))) * (-ellipk(m[~mask]) + ((self.radius**2 + R[~mask]**2 + Z[~mask]**2)/q[~mask]) * ellipe(m[~mask]))
            b_r_val[near_axis] = (3 * mu_0 * self.radius**2 * Z[near_axis] * R[near_axis])/(4 * (self.radius**2 + Z[near_axis]**2)**(5/2))

            return b_r_val

        def b_r_shifted(R, Z):
            return b_r(R, Z - self.z_pos)

        self.b_r = b_r_shifted


    def _create_b_z(self):

        def b_z(R, Z):
            '''
            R and Z expected to be meshgrids created by np.meshgrid(R, Z).
            Returns grid in same shape as R and Z with values of b_z filled in
            '''
            D = (self.radius + R)**2 + Z**2
            m = (4 * self.radius * R)/D
            q = (self.radius - R)**2 + Z**2

            near_wire = (R - self.radius)**2 + (Z)**2 < self.wire_radius**2

            b_z_val = np.full_like(R, np.nan, dtype=float)

            b_z_val[~near_wire] = (mu_0/(2 * pi * np.sqrt(D[~near_wire]))) * (ellipk(m[~near_wire]) + ((self.radius**2 - R[~near_wire]**2 - Z[~near_wire]**2)/q[~near_wire]) * ellipe(m[~near_wire]))

            return b_z_val

        def b_z_shifted(R, Z):
            return b_z(R, Z - self.z_pos)

        self.b_z = b_z_shifted