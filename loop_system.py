import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.constants import mu_0

class loop_system:
    '''
    Each loop in loops needs to have a unique name, otherwise there could be some issues...
    '''

    def __init__(self, loops):
        '''
        loops:                  list of current_loop objects
        background_Bfield_z:    float for strength [T] of background field
        '''
        self.loops = loops

    def total_Efield_phi(self, t, R, Z):
        '''
        Calculates E_phi(t, R, Z). t is an array of times, R and Z come from np.meshgrid. Returns an array
        where array[t] gives a grid that is the same size as R and Z with Efield_phi filled out.
        '''

        Efield_phi_vals = np.full((len(t), R.shape[0], R.shape[1]), 0.0)
        
        for loop in self.loops:
            Efield_phi_vals += loop.Efield_phi(R, Z, t)

        return Efield_phi_vals

    def total_Bfield_R(self, t, R, Z):
        '''
        Calculates B_R(t, R, Z). t is an array of times, R and Z come from np.meshgrid. Returns an array
        where array[t] gives a grid that is the same size as R and Z with Bfield_R filled out.
        '''
        Bfield_R_vals = np.full((len(t), R.shape[0], R.shape[1]), 0.0)

        for loop in self.loops:
            Bfield_R_vals += loop.Bfield_R(R, Z, t)

        return Bfield_R_vals
    
    def total_Bfield_Z(self, t, R, Z):
        '''
        Calculates B_Z(t, R, Z). t is an array of times, R and Z come from np.meshgrid. Returns an array
        where array[t] gives a grid that is the same size as R and Z with Bfield_Z filled out.
        '''
        Bfield_Z_vals = np.full((len(t), R.shape[0], R.shape[1]), 0.0)

        for loop in self.loops:
            Bfield_Z_vals += loop.Bfield_Z(R, Z, t)

        return Bfield_Z_vals

    def calc_inductance_mtrx(self):
        '''
        Calculates the mutual inductance matrix, [M] where M_ij = flux through I per current in J
        '''
        num_loops = len(self.loops)
        induct_mtrx = np.full((num_loops, num_loops), np.nan)

        for i in range(num_loops):
            for j in  range(num_loops):
                induct_mtrx[i][j] = self.calc_mutual_induc(self.loops[i], self.loops[j])

        return induct_mtrx

    def calc_mutual_induc(self, first_loop, second_loop):
        '''
        Calculates flux through first_loop per current in second_loop
        '''
        l1_zpos = first_loop.z_pos
        l1_rad  = first_loop.radius
        l2_zpos = second_loop.z_pos
        l2_rad  = second_loop.radius

        if (l1_zpos == l2_zpos) and (l1_rad == l2_rad):
            return mu_0 * l1_rad * (np.log(8 * (l1_rad/first_loop.wire_radius)) - 2)

        R, Z = np.meshgrid(l1_rad, l1_zpos)

        return second_loop.a_phi(R, Z)[0][0] * 2 * np.pi * l1_rad

    # ===================== ANIMATIONS =====================
    def animate_loop_voltage(self, radius, t_start, t_stop, frames, fps, min_z, max_z, z_res, r_max, r_res):
        '''
        animates v_loop_r(z, t) = E_phi(r, z , t) * 2pi * r. 
        '''

        Z = np.linspace(min_z, max_z, z_res)
        R = np.linspace(0, r_max, r_res)
        times = np.linspace(t_start, t_stop, frames)
        rad_indx = np.argmin(np.abs(R - radius))

        R_grid, Z_grid = np.meshgrid(R, Z)

        Efield_vals = self.total_Efield_phi(times, R_grid, Z_grid)
        voltages = Efield_vals[:, :, rad_indx] * 2 * np.pi * R[rad_indx]

        # Animation stuff
        fig, ax = plt.subplots(figsize=(8, 5))

        (line, ) = ax.plot(Z, voltages[0])
        ax.set_xlim(np.min(Z), np.max(Z))
        ax.set_ylim(np.nanmin(voltages), np.nanmax(voltages))

        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Volts")
        title = ax.set_title('Voltage vs Z at t = 0.0')

        def update(frame_index):
            line.set_ydata(voltages[frame_index])
            title.set_text(f"Voltage vs Z at t = {times[frame_index]} (at R = {R[rad_indx]})")

            return line, title

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,  # Number of frames equals the number of time steps
            interval= 1000 / fps,  # Delay between frames in milliseconds (lower = faster)
            blit=True,  # Optimization for smooth rendering
        )  

        plt.close(fig)

        return ani





