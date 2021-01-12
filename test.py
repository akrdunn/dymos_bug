import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.plotting import plot_results
from brachistochrone_ode import BrachistochroneODE
import matplotlib.pyplot as plt
import numpy as np

#
# Initialize the Problem and the optimization driver
#
p = om.Problem(model=om.Group())
p.driver = om.ScipyOptimizeDriver()
p.driver.declare_coloring()

#
# Create a trajectory and add a phase to it
#
traj = p.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=BrachistochroneODE,
                                transcription=dm.GaussLobatto(num_segments=10)))

#
# Set the variables
#
phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot')

phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot')

phase.add_state('v', fix_initial=True, fix_final=False, rate_source='vdot')

phase.add_control('theta', continuity=True, rate_continuity=True,
                  units='deg', lower=0.01, upper=179.9)

phase.add_parameter('g', units='m/s**2', val=9.80665)

# dummy array of data
indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('array', np.linspace(1, 10, 10), units=None)
# add dummy array as a parameter and connect it
phase.add_parameter('array', units=None, shape=(10,), dynamic=False)
p.model.connect('array', 'traj.phase0.parameters:array')

#
# Minimize time at the end of the phase
#
phase.add_objective('time', loc='final', scaler=10)

p.model.linear_solver = om.DirectSolver()

#
# Setup the Problem
#
p.setup()

#
# Set the initial values
#
p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = 2.0

p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
p['traj.phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

#
# Solve for the optimal trajectory
#
dm.run_problem(p)

# Test the results


# Generate the explicitly simulated trajectory
exp_out = traj.simulate()

plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
               'x (m)', 'y (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:theta',
               'time (s)', 'theta (deg)')],
             title='Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method',
             p_sol=p, p_sim=exp_out)

plt.show()