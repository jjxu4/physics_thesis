import numpy as np

class loop_system:

    def __init__(self, loops, background_Bfield_z):
        '''
        loops:                  list of current_loop objects
        background_Bfield_z:    float for strength [T] of background field
        '''
        self.loops = loops
        self.background_Bfield_z = background_Bfield_z

    def total_Efield_phi(self, t, R, Z):
        '''
        Calculates E_phi(t, R, Z). t is an array of times, R and Z come from np.meshgrid. Returns an array
        where array[t] gives a grid that is the same size as R and Z with Efield_phi filled out.
        '''

        Efield_phi_vals = np.full((len(t), R.shape[0], R.shape[1]), 0)

        for time in t:
            # sum over Efield_phi from each loop
            for loop in self.loops:
                Efield_phi_vals[time] += loop.Efield_phi(R, Z)

        return Efield_phi_vals

    def total_Bfield_R(self, t, R, Z):
        '''
        Calculates B_R(t, R, Z). t is an array of times, R and Z come from np.meshgrid. Returns an array
        where array[t] gives a grid that is the same size as R and Z with Bfield_R filled out.
        '''
        Bfield_R_vals = np.full((len(t), R.shape[0], R.shape[1]), 0)

        for time in t:
            # sum over Bfield_R from each loop
            for loop in self.loops:
                Bfield_R_vals[time] += loop.Bfield_R(R, Z)

        return Bfield_R_vals
    
    def total_Bfield_Z(self, t, R, Z):
        '''
        Calculates B_Z(t, R, Z). t is an array of times, R and Z come from np.meshgrid. Returns an array
        where array[t] gives a grid that is the same size as R and Z with Bfield_Z filled out.
        '''
        Bfield_Z_vals = np.full((len(t), R.shape[0], R.shape[1]), 0)

        for time in t:
            # sum over Bfield_R from each loop
            for loop in self.loops:
                Bfield_Z_vals[time] += loop.Bfield_Z(R, Z)

        return Bfield_Z_vals


    # ===================== ANIMATIONS =====================
    def animate_loop_voltage(self):
