import numpy as np
from scipy import interpolate

def smooth_trajectory(time_steps_raw, positions_raw, num_actions, num_joints):
        duration = time_steps_raw[-1]

        smooth_steps = np.linspace(0, duration, num_actions)

        spls = [interpolate.splrep(time_steps_raw, positions_raw[:,i])
                                               for i in range(num_joints)]

        smooth_positions = np.stack([interpolate.splev(smooth_steps, spls[i], der=0) for i in range(num_joints)])
        smooth_velocities = np.stack([interpolate.splev(smooth_steps, spls[i], der=1) for i in range(num_joints)]).T
        smooth_accelerations = np.stack([interpolate.splev(smooth_steps, spls[i], der=2) for i in range(num_joints)]).T

        return smooth_steps, smooth_positions, smooth_velocities, smooth_accelerations

MAX_ANGLE = np.pi
MIN_ANGLE = -np.pi
