from __future__ import annotations
import os
import cloudpickle as cp
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from sympy import Symbol, sqrt, Matrix, cos, pi, ImmutableMatrix, Sum, IndexedBase, symbols, Piecewise, GreaterThan
from opty.direct_collocation import Problem
from dataclasses import replace
from src.Optimization.opt_container import DataStorage, SteerWith, ConstraintStorage, SeatType, TorsoType, Task
from src.Optimization.opt_utils import create_objective_function, plot_constraint_violations
from src.Optimization.opt_model import set_model, SphericalTorsoJoint, PinTorsoJoint, FixedTorsoJoint, NeckPinJoint, FixedNeck
from src.Optimization.opt_model import SideLeanSeat, ShiftingSideLeanSeat, InterSeatJoint

# Corrected version from opty's ``Problem.plot_constraint_violations``.
plot_constraint_violations.__doc__ = Problem.plot_constraint_violations.__doc__
Problem.plot_constraint_violations = plot_constraint_violations

def set_constraints(data: DataStorage) -> None:
    t = me.dynamicsymbols._t  # Time symbol.
    # Initial and final time.
    t1, tf = data.metadata.duration / data.metadata.num_nodes, data.metadata.duration
    bicycle, rider = data.bicycle, data.rider
    bicycle_rider = data.bicycle_rider

    initial_state_constraints = {
        bicycle.q[0]: 0.0,
        bicycle.q[1]: 0.0,
        bicycle.q[2]: 0.0,
        bicycle.q[3]: 0.0,
        bicycle.q[4]: 0.0,
        bicycle.q[5]: 0.0,
        bicycle.q[6]: 0.0,
        bicycle.q[7]: 0.0,
        bicycle.u[1]: 0.0,
        bicycle.u[2]: 0.0,
        bicycle.u[3]: 0.0,
        bicycle.u[4]: 0.0,
        bicycle.u[6]: 0.0,
    }

    if data.metadata.model_upper_body:
        print("if model_upper_body caller werkt wel")
        if data.metadata.seat_type == data.metadata.seat_type.SIDELEAN:
            print(".")
            print("The leaning seat ICSs should be added now.")
            initial_state_constraints[bicycle_rider.seat.q[0]] = np.deg2rad(0.0)
            initial_state_constraints[bicycle_rider.seat.u[0]] = 0.0
        elif data.metadata.seat_type == data.metadata.seat_type.SHIFTINGSIDELEAN:
            print('the shifting seat ISCs should work atm.')
            initial_state_constraints[bicycle_rider.seat.q[0]] = np.deg2rad(0.0)
            initial_state_constraints[bicycle_rider.seat.u[0]] = 0.0
            initial_state_constraints[rider.shiftingsideleanseat.q[0]] = np.deg2rad(0.0)
            initial_state_constraints[rider.shiftingsideleanseat.u[0]] = 0.0
    if data.metadata.model_torso:
        if type(rider.torsojoint) == PinTorsoJoint:
            initial_state_constraints[rider.torsojoint.q[0]] = np.deg2rad(0.0)
            initial_state_constraints[rider.torsojoint.u[0]] = 0.0
        elif type(rider.torsojoint) == SphericalTorsoJoint:
            initial_state_constraints[rider.torsojoint.q[0]] = np.deg2rad(0.0)
            initial_state_constraints[rider.torsojoint.u[0]] = 0.0
            initial_state_constraints[rider.torsojoint.q[1]] = np.deg2rad(0.0)
            initial_state_constraints[rider.torsojoint.u[1]] = 0.0
            initial_state_constraints[rider.torsojoint.q[2]] = np.deg2rad(0.0)
            initial_state_constraints[rider.torsojoint.u[2]] = 0.0
        elif type(rider.torsojoint) == FixedTorsoJoint:
            None
        if data.metadata.model_head:
            if type(rider.neck) == NeckPinJoint:
                initial_state_constraints[rider.neck.q[0]] = np.deg2rad(0)  # adduction
                initial_state_constraints[rider.neck.u[0]] = 0
            elif type(rider.neck) == FixedNeck:
                None

    if data.metadata.model_legs:
        initial_state_constraints[rider.left_hip.q[0]] = np.deg2rad(70)     # flexion
        #initial_state_constraints[rider.left_hip.u[0]] = np.deg2rad(0)
        initial_state_constraints[rider.left_hip.q[1]] = np.deg2rad(0)      # adduction
        #initial_state_constraints[rider.left_hip.u[1]] = np.deg2rad(0)
        initial_state_constraints[rider.left_hip.q[2]] = np.deg2rad(0)      # rotation
        #initial_state_constraints[rider.left_hip.u[2]] = np.deg2rad(0)
        initial_state_constraints[rider.right_hip.q[0]] = np.deg2rad(70)
        #initial_state_constraints[rider.right_hip.u[0]] = np.deg2rad(0)
        initial_state_constraints[rider.right_hip.q[1]] = np.deg2rad(0)
        #initial_state_constraints[rider.right_hip.u[1]] = np.deg2rad(0)
        initial_state_constraints[rider.right_hip.q[2]] = np.deg2rad(0)
        #initial_state_constraints[rider.right_hip.u[2]] = np.deg2rad(0)
        """
        initial_state_constraints[rider.left_leg.q[0]] = np.deg2rad(-45)
        initial_state_constraints[rider.left_leg.u[0]] = np.deg2rad(0)
        initial_state_constraints[rider.left_leg.q[1]] = np.deg2rad(-30)
        initial_state_constraints[rider.left_leg.u[1]] = np.deg2rad(0)
        initial_state_constraints[rider.right_leg.q[0]] = np.deg2rad(-105)
        initial_state_constraints[rider.right_leg.u[0]] = np.deg2rad(0)
        initial_state_constraints[rider.right_leg.q[1]] = np.deg2rad(30)
        initial_state_constraints[rider.right_leg.u[1]] = np.deg2rad(0)
        """
    print('los initial stato constrantos son:', initial_state_constraints)

    match data.metadata.task:
        case data.metadata.task.LANE_SWITCH:
            print('de task.lane_switch zou nu FSCs moeten hebben')
            final_state_constraints = {
                bicycle.q[0]: 3 * data.metadata.straight_length,
                bicycle.q[1]: data.metadata.lateral_displacement,
                bicycle.q[2]: np.deg2rad(0),
                bicycle.q[3]: 0.0,
                bicycle.q[6]: 0.0,
            }
        case data.metadata.task.DOUBLE_LANE_SWITCH:
            print('de task.double_lane_switch zou nu FSCs moeten hebben')
            final_state_constraints = {
                bicycle.q[0]: 5 * data.metadata.straight_length,
                bicycle.q[1]: 0,
                bicycle.q[2]: np.deg2rad(0),
                bicycle.q[3]: 0.0,
                bicycle.q[6]: 0.0,
            }
        case data.metadata.task.STRAIGHT_TURN:
            print('de task.straight_turn zou nu FSCs moeten hebben')
            final_state_constraints = {
                bicycle.q[0]: data.metadata.straight_length + data.metadata.turn_radius,
                bicycle.q[1]: data.metadata.straight_length + data.metadata.turn_radius,
                bicycle.q[2]: np.deg2rad(90),
                bicycle.q[3]: 0.0,
                bicycle.q[6]: 0.0,
            }
        case data.metadata.task.PERTURBED_CYCLING:
            print('de task.perturbed_cycling zou nu FSCs moeten hebben')
            final_state_constraints = {
                bicycle.q[0]: 5 * data.metadata.straight_length,
                bicycle.q[1]: 0,
                bicycle.q[2]: np.deg2rad(0),
                bicycle.q[3]: 0.0,
                #bicycle.q[6]: 0.0,
            }
            
        case data.metadata.task.MATCH_EXP_DATA:
            print('task.MATCH_EXP_DATA should now have FSCs')
            final_state_constraints = {
                bicycle.q[0]: 5 * data.metadata.straight_length,
                bicycle.q[1]: 0,
                bicycle.q[2]: np.deg2rad(0),
                bicycle.q[3]: 0.0,
                #bicycle.q[6]: 0.0,
            }     
            
            
    print('biycle FCSs', final_state_constraints)

    if data.metadata.model_upper_body:
        if data.metadata.seat_type == data.metadata.seat_type.SIDELEAN:
            print('normal sideleanseat final state constraints added')
            final_state_constraints.update({
                bicycle_rider.seat.q[0]: np.deg2rad(0.0),
                bicycle_rider.seat.u[0]: 0.0})
        elif data.metadata.seat_type == data.metadata.seat_type.SHIFTINGSIDELEAN:
            print('the shifting sidelean seat final state constraints added')
            final_state_constraints.update({
                bicycle_rider.seat.q[0]: 0.0,
                bicycle_rider.seat.u[0]: 0.0,
                rider.shiftingsideleanseat.q[0]: np.deg2rad(0.0),
                rider.shiftingsideleanseat.u[0]: 0.0})
    if data.metadata.model_torso:
        if type(rider.torsojoint) == PinTorsoJoint:
            final_state_constraints.update({
                rider.torsojoint.q[0]: 0.0,
                rider.torsojoint.u[0]: 0.0})
        elif type(rider.torsojoint) == SphericalTorsoJoint:
            final_state_constraints.update({
                rider.torsojoint.q[0]: 0.0,
                rider.torsojoint.u[0]: 0.0,
                rider.torsojoint.q[1]: 0.0,
                rider.torsojoint.u[1]: 0.0,
                rider.torsojoint.q[2]: 0.0,
                rider.torsojoint.u[2]: 0.0})
        elif type(rider.torsojoint) == FixedTorsoJoint:
            None
        if data.metadata.model_head:
            if type(rider.neck) == FixedNeck:
                None
            elif type(rider.neck) == NeckPinJoint:
                final_state_constraints.update({
                    rider.neck.q[0]: 0.0,
                    rider.neck.u[0]: 0.0})

#    if data.metadata.model_legs:   # i don't think i need to end-constrain these, the legs can do whatever they want
#        final_state_constraints.update({
#            rider.left_leg.q[1]: 1.31,
#            rider.right_leg.q[1]: 1.83})
#        final_state_constraints.update({})

    instance_constraints = tuple(
        xi.replace(t, t1) - xi_val for xi, xi_val in initial_state_constraints.items()
        ) + tuple(
        xi.replace(t, tf) - xi_val for xi, xi_val in final_state_constraints.items())


    if data.metadata.task == data.metadata.task.LANE_SWITCH:
        bounds = {
            bicycle.q[0]: (-0.2, 3 * data.metadata.straight_length + 0.2),
            bicycle.q[1]: (-0.4, data.metadata.lateral_displacement + 0.4)}
    elif data.metadata.task == data.metadata.task.DOUBLE_LANE_SWITCH:
        bounds = {
            bicycle.q[0]: (-0.2, 5 * data.metadata.straight_length + 0.2),
            bicycle.q[1]: (-0.4, data.metadata.lateral_displacement + 0.4)}
    elif data.metadata.task == data.metadata.task.STRAIGHT_TURN:
        bounds = {
            bicycle.q[0]: (-0.2, data.metadata.straight_length + data.metadata.turn_radius + 0.2),
            bicycle.q[1]: (-0.2, data.metadata.straight_length + data.metadata.turn_radius + 0.2)}
    elif data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        bounds = {
            bicycle.q[0]: (-0.2, 5 * data.metadata.straight_length + 0.2),
            bicycle.q[1]: (-1.5, 1.5)}
    elif data.metadata.task == data.metadata.task.MATCH_EXP_DATA:
        bounds = {
            bicycle.q[0]: (-0.2,  data.metadata.straight_length ), # JR
            bicycle.q[1]: (-3, 3)} #JR

    bounds.update({
        bicycle.q[2]: (-np.deg2rad(180), np.deg2rad(180)),  # bicycle yaw
        bicycle.q[3]: (-np.deg2rad(45), np.deg2rad(45)),    # bicycle roll
        bicycle.q[4]: (np.deg2rad(10), np.deg2rad(40)),     # bicycle pitch
        bicycle.q[5]: (-200.0, 10.0),                       # rear wheel
        bicycle.q[6]: (-np.deg2rad(70), np.deg2rad(70)),    # steering
        bicycle.q[7]: (-200.0, 10.0),
        bicycle.u[0]: (-2, 10.0),
        bicycle.u[1]: (-10.0, 10.0),
        bicycle.u[2]: (-5.0, 5.0),
        bicycle.u[3]: (-2.5, 2.5),
        bicycle.u[4]: (-1.0, 1.0),
        bicycle.u[5]: (-30.0, 5.0),
        bicycle.u[6]: (-3.0, 3.0),
        bicycle.u[7]: (-30.0, 5.0)})

    if data.metadata.model_upper_body:
        if data.metadata.seat_type == data.metadata.seat_type.SIDELEAN:
            bounds.update({
                bicycle_rider.seat.q[0]: (np.deg2rad(-30), np.deg2rad(30)),
                bicycle_rider.seat.u[0]: (np.deg2rad(-45), np.deg2rad(45))})
        elif data.metadata.seat_type == data.metadata.seat_type.SHIFTINGSIDELEAN:
            bounds.update({
                rider.shiftingsideleanseat.q[0]: (np.deg2rad(-30), np.deg2rad(30)),
                rider.shiftingsideleanseat.u[0]: (np.deg2rad(-45), np.deg2rad(45)),
                bicycle_rider.seat.q[0]: (-1, 1),   ## meters i assume
                bicycle_rider.seat.u[0]: (-5.0, 5.0)})
        if data.metadata.model_torso:
            print('adding the torsojoint bounds')
            if type(rider.torsojoint) == PinTorsoJoint:
                bounds.update({
                    rider.torsojoint.q[0]: (np.deg2rad(-60), np.deg2rad(60)),
                    rider.torsojoint.u[0]: (np.deg2rad(-90), np.deg2rad(90))})
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                bounds.update({
                    rider.torsojoint.q[0]: (np.deg2rad(-20), np.deg2rad(15)),
                    rider.torsojoint.u[0]: (np.deg2rad(-90), np.deg2rad(90)),
                    rider.torsojoint.q[1]: (np.deg2rad(-60), np.deg2rad(60)),
                    rider.torsojoint.u[1]: (np.deg2rad(-90), np.deg2rad(90)),
                    rider.torsojoint.q[2]: (np.deg2rad(-30), np.deg2rad(30)),
                    rider.torsojoint.u[2]: (np.deg2rad(-90), np.deg2rad(90))})
            elif type(rider.torsojoint) == FixedTorsoJoint:
                None
            if data.metadata.model_head:
                print('adding the neck bounds if needed')
                if type(rider.neck) == NeckPinJoint:
                    bounds.update({
                        rider.neck.q[0]: (np.deg2rad(-45), np.deg2rad(45)),
                        rider.neck.u[0]: (np.deg2rad(-60), np.deg2rad(60))})
                elif type(rider.neck) == FixedNeck:
                    None
        if data.metadata.model_legs:
            print('adding the hip&leg bounds')
            bounds.update({
                rider.left_leg.q[0]: (np.deg2rad(-135), 0),   # knee
                rider.left_leg.u[0]: (-5, 5),
                rider.left_leg.q[1]: (np.deg2rad(-30), np.deg2rad(30)),  # ankle
                rider.left_leg.u[1]: (-5, 5),
                rider.right_leg.q[0]: (np.deg2rad(-135), 0),
                rider.right_leg.u[0]: (-5, 5),
                rider.right_leg.q[1]: (np.deg2rad(-30), np.deg2rad(30)),  # ankle
                rider.right_leg.u[1]: (-5, 5),

                rider.left_hip.q[0]: (np.deg2rad(25), np.deg2rad(60)),   # flexion
                rider.left_hip.u[0]: (-2, 2),
                rider.left_hip.q[1]: (np.deg2rad(-45), np.deg2rad(45)),     # adduction
                rider.left_hip.u[1]: (-4, 4),
                rider.left_hip.q[2]: (np.deg2rad(-45), np.deg2rad(45)),   # rotation
                rider.left_hip.u[2]: (-4, 4),

                rider.right_hip.q[0]: (np.deg2rad(25), np.deg2rad(60)),  # flexion
                rider.right_hip.u[0]: (-2, 2),
                rider.right_hip.q[1]: (np.deg2rad(-45), np.deg2rad(45)),    # adduction
                rider.right_hip.u[1]: (-4, 4),
                rider.right_hip.q[2]: (np.deg2rad(-45), np.deg2rad(45)),   # rotation
                rider.right_hip.u[2]: (-4, 4)
            })

    print('length bounds:', len(bounds), ', and the length of the init_state_constraints:', len(initial_state_constraints))
    print('input vars before setting bounds:', data.input_vars)

    if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
        bounds.update({
            data.input_vars[0]: (-100.0, 100.0),   # seat torque
        })
        if data.metadata.sprung_steering == True and data.fixed_stiffness == False:
            bounds.update({
                data.input_vars[1]: (-20.0, 20.0)})
            print('the bounds after supposedly adding bounds for the variable spring:', bounds)
    elif data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
        if type(rider.torsojoint) == PinTorsoJoint:
            bounds.update({
                data.input_vars[0]: (-1000.0, 1000.0),  # torsojoint torque
            })
            if data.metadata.sprung_steering == True and data.fixed_stiffness == False:
                bounds.update({
                    data.input_vars[1]: (-20.0, 20.0)})
        if type(rider.torsojoint) == SphericalTorsoJoint:
            bounds.update({
                data.input_vars[0]: (-500.0, 500.0),
                data.input_vars[1]: (-500.0, 500.0),
                data.input_vars[2]: (-500.0, 500.0),
            })
            if data.metadata.sprung_steering == True and data.fixed_stiffness == False:
                bounds.update({
                    data.input_vars[3]: (-20.0, 20.0)})
    elif data.metadata.steer_with == data.metadata.steer_with.SEAT_AND_TORSO_TORQUE:
        bounds.update({
            data.input_vars[0]: (-500, 500),
            data.input_vars[1]: (-500, 500)})
        if data.metadata.sprung_steering == True and data.fixed_stiffness == False:
            bounds.update({
                data.input_vars[2]: (-20.0, 20.0)})
    elif data.metadata.steer_with == data.metadata.steer_with.LEG_TORQUE:
        bounds.update({
            data.input_vars[0]: (-20, 20),  # left hip adduction
            data.input_vars[1]: (-30, 30),  # left hip flexion
            data.input_vars[2]: (-4, 4),    # left hip rotation
            data.input_vars[3]: (-10, 4),   # left ankle
            data.input_vars[4]: (10, 10),   # left knee
            data.input_vars[5]: (-20, 20),  # right hip adduction
            data.input_vars[6]: (-30, 30),  # right hip flexion   ## these aren't correct anymore as the leg torques have been canceled
            data.input_vars[7]: (-4, 4),    # right hip rotation
            data.input_vars[8]: (-10, 4),   # right ankle
            data.input_vars[9]: (-10, 10)   # right knee
        })
    elif data.metadata.steer_with == data.metadata.steer_with.UPPER_BODY_TORQUE:
        bounds.update({
            data.input_vars[0]: (-500, 500),  # neck torque
            data.input_vars[1]: (-500, 500),    # pedal torque
            data.input_vars[2]: (-500, 500)})  # seat torque
        if data.metadata.sprung_steering == True and data.fixed_stiffness == False:
            bounds.update({
                data.input_vars[3]: (-20.0, 20.0)})

    ### this part below is to see if the RMS error can be set to a target for the optimizer
    s = data.metadata.straight_length
    d_lat = data.metadata.lateral_displacement
    path_length = 5 * s
    path_len = np.linspace(0, path_length, data.metadata.num_nodes)
    d_long1 = 3 * s
    d_long2 = 5 * s
    q1_path = []
    q2_path = []

    for q1 in path_len:
        if q1 < s:
            q2 = 0
        elif q1 <= d_long1 - s:
            progress = (q1 - s) / (d_long1 - 2 * s)
            q2 = d_lat * (1 - np.cos(np.pi * progress)) / 2
        elif q1 <= d_long1:
            q2 = d_lat
        elif q1 <= d_long2 - s:
            progress = (q1 - d_long1) / (d_long1 - 2 * s)
            q2 = d_lat * (1 + np.cos(np.pi * progress)) / 2
        else:
            q2 = 0
        q1_path.append(q1)
        q2_path.append(q2)

    q1_path = np.array(q1_path)
    q2_path = np.array(q2_path)
    q1_sym = Matrix(q1_path)  # make it a sympy matrix
    q2_sym = Matrix(q2_path)
    # Calculate the errors for q1 and q2 paths relative to the bike's current state
    q1_error = q1_sym - Matrix([bicycle.q[0]] * len(q1_path))  # make the symbolic variable
    q2_error = q2_sym - Matrix([bicycle.q[1]] * len(q2_path))

    q1_error_squared = q1_error.applyfunc(lambda x: x ** 2)
    q2_error_squared = q2_error.applyfunc(lambda x: x ** 2)
    combined_error = q1_error_squared + q2_error_squared
    time_vector = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)

    ms_err_expr = sum(combined_error) / combined_error.shape[0]
    ms_err = ms_err_expr.simplify()

    data.ms_error = ms_err


    def function(x):
        """ Silly function that has a nice dip at x = 0.04"""
        return 370 - 200 * (1 / (1 + sm.exp(1 * (x - 2.08))) + 1 / (1 + sm.exp(-1  *(x + 2))))

    #ms_goal = (function(ms_err)*function)
    #print('ms goal:', ms_goal)

    constrain_rms = False
    ## back to the actually used code:
    print('control weight:', data.metadata.weight_ct, 'tracking weight:', data.metadata.weight_tr, 'verticality weight:', (1 - data.metadata.weight_ct - data.metadata.weight_tr))
    if data.metadata.task == data.metadata.task.LANE_SWITCH:
        print('should make the objective_expr of the lane_switch now')
        data.objective_expr = (
            data.metadata.weight * (data.lane_switch_task) ** 2 +
            (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))
    elif data.metadata.task == data.metadata.task.DOUBLE_LANE_SWITCH and constrain_rms == True:
        print('should make the objective_expr of the double lane_switch now')
        data.objective_expr = ((1 * sum(i ** 2 for i in data.input_vars))
        + function(ms_err)) #+ (data.lane_switch_task) ** 2

    elif data.metadata.task == data.metadata.task.DOUBLE_LANE_SWITCH and constrain_rms == False:
        print('should make the objective_expr of the double lane_switch now')
        data.objective_expr = (
            data.metadata.weight * (data.double_lane_switch_task) ** 2 +
            (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))
    elif data.metadata.task == data.metadata.task.STRAIGHT_TURN:
        print('should make the objective_expr of the straight_turn now')
        data.objective_expr = (
            data.metadata.weight * (data.straight_turn_task) ** 2 +
            (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))
    elif data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        print('should make the objective_expr of the perturbed cycling now')
        data.objective_expr = (
            data.metadata.weight * (data.straight_line_task) ** 2 +
            (1 - data.metadata.weight) * sum(i ** 2 for i in data.input_vars))
    elif data.metadata.task == data.metadata.task.MATCH_EXP_DATA: #JR
        print('should make the objective_expr of the experimental data matching')
        data.objective_expr = data.match_exp_data_task **2

    data.constraints = ConstraintStorage(
        initial_state_constraints, final_state_constraints, instance_constraints, bounds)

    print('constraints should be set now.')
    print('bounds: ', bounds)
    print('length & initial state constraints: ', len(initial_state_constraints), initial_state_constraints)
    print('final state constraints: ', final_state_constraints)

def set_problem(data: DataStorage) -> None:
    print('data.objective_expr:', data.objective_expr)
    print('the input var data now is:', data.input_vars)
    obj, obj_grad = create_objective_function(data, data.objective_expr)
    print(len(data.constants), len(data.initial_guess), len(data.eoms), len(data.system.loads), len(data.system.kdes))
    print('the constants are --->', data.constants)
    #new_instance_constraints = data.constraints.instance_constraints + (data.ms_error <= 0.040,)
    #data.constraints = replace(data.constraints, instance_constraints=new_instance_constraints)

    if data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        problem = Problem(
            obj,
            obj_grad,
            data.eoms,
            data.x,
            data.metadata.num_nodes,
            data.metadata.interval_value,
            known_trajectory_map={data.wind: data.wind_array},
            known_parameter_map=data.constants,
            instance_constraints=data.constraints.instance_constraints,
            bounds=data.constraints.bounds,
            integration_method='backward euler')
    else:
        problem = Problem(
            obj,
            obj_grad,
            data.eoms,
            data.x,
            data.metadata.num_nodes,
            data.metadata.interval_value,
            known_parameter_map=data.constants,
            instance_constraints=data.constraints.instance_constraints,
            bounds=data.constraints.bounds,
            integration_method='backward euler')

    #problem.add_option('nlp_scaling_method', 'gradient-based')
    data.problem = problem
    print('problem:', problem)


def set_initial_guess(data: DataStorage) -> None:
    print('obj_expr,, constraints,, in the set_initial_guess:', data.objective_expr, data.constraints)
    d_lat = data.metadata.lateral_displacement
    radius = data.metadata.turn_radius
    d_straight = data.metadata.straight_length
    task_speed = 15 / 3.6  # from kmh to m/s

    if data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        length = 5 * data.metadata.straight_length
        vel_mean = length / data.metadata.duration
    elif data.metadata.task == data.metadata.task.LANE_SWITCH:
        length = (2 * d_straight + np.sqrt(d_lat**2 + d_straight**2))
        vel_mean = length / data.metadata.duration
    elif data.metadata.task == data.metadata.task.DOUBLE_LANE_SWITCH:
        length = (3 * d_straight + 2 * np.sqrt(d_lat**2 + d_straight**2))
        vel_mean = length / data.metadata.duration
    elif data.metadata.task == data.metadata.task.STRAIGHT_TURN:
        length = (2 * d_straight + np.pi * radius / 2)
        vel_mean = length / data.metadata.duration
        
        
        
        
        
    print('track length:', length, '[m]')
    print('vel_mean:', vel_mean, '[m/s]')
    print('preferred vel_mean duration:', length / task_speed, '[s]')
    angle = 0.0
    q2_0 = 0.0
    q1_0 = 0.0
    rr = data.constants[data.bicycle.rear_wheel.radius]

    data.simulator.initial_conditions = {
        **{xi: data.constraints.initial_state_constraints.get(xi, 0.0)
           for xi in data.x},
        data.bicycle.u[0]: vel_mean, # * np.cos(angle),
        data.bicycle.u[1]: 0.0, #vel_mean * np.sin(angle),
        data.bicycle.q[0]: q1_0,
        data.bicycle.q[1]: q2_0,
        data.bicycle.q[2]: angle,
        data.bicycle.u[5]: -vel_mean / rr,
        data.bicycle.u[7]: -vel_mean / rr}
    print('initial guess x0 bicycle added: ', data.simulator.initial_conditions)
    if data.metadata.model_upper_body:
        if data.metadata.seat_type == data.metadata.seat_type.SIDELEAN:
            data.simulator.initial_conditions = {
                **data.simulator.initial_conditions,
                data.bicycle_rider.seat.q[0]: 0.0,
                data.bicycle_rider.seat.u[0]: 0.0}
            print('has a normal leaning seat x0')
        elif data.metadata.seat_type == data.metadata.seat_type.SHIFTINGSIDELEAN:
            data.simulator.initial_conditions = {
                **data.simulator.initial_conditions,
                data.bicycle_rider.seat.q[0]: 0.0,
                data.bicycle_rider.seat.u[0]: 0.0,
                data.rider.shiftingsideleanseat.q[0]: 0.0,
                data.rider.shiftingsideleanseat.u[0]: 0.0}
            print('should have added shiftingseat x0')
        elif data.metadata.seat_type == data.metadata.seat_type.FIXED:
            None
    if data.metadata.model_torso:
        if type(data.rider.torsojoint) == PinTorsoJoint:
            data.simulator.initial_conditions = {
                **data.simulator.initial_conditions,
                data.rider.torsojoint.q[0]: 0.0,
                data.rider.torsojoint.u[0]: 0.0}
            print('should have added pintorsojoint x0')
        elif type(data.rider.torsojoint) == SphericalTorsoJoint:
            data.simulator.initial_conditions = {
                **data.simulator.initial_conditions,
                data.rider.torsojoint.q[0]: 0.0,
                data.rider.torsojoint.u[0]: 0.0,
                data.rider.torsojoint.q[1]: 0.0,
                data.rider.torsojoint.u[1]: 0.0,
                data.rider.torsojoint.q[2]: 0.0,
                data.rider.torsojoint.u[2]: 0.0}
            print('spherical torsojoint x0 added')
        elif type(data.rider.torsojoint) == FixedTorsoJoint:
            None
        if data.metadata.model_head:
            if type(data.rider.neck) == NeckPinJoint:
                data.simulator.initial_conditions = {
                    **data.simulator.initial_conditions,
                    # Some initial guesses for the torsojoint flexion, lean and twisting angle.
                    data.rider.neck.q[0]: 0.0,
                    data.rider.neck.u[0]: 0.0}
                print('should have added the neck initial guess')
            elif type(data.rider.neck) == FixedNeck:
                None
    if data.metadata.model_legs:
        data.simulator.initial_conditions = {
            **data.simulator.initial_conditions,
            data.rider.left_hip.q[0]: np.deg2rad(30),
            data.rider.left_hip.u[0]: 0.0,
            data.rider.left_hip.q[1]: 0.0,
            data.rider.left_hip.u[1]: 0.0,
            data.rider.left_hip.q[2]: 0.0,
            data.rider.left_hip.u[2]: 0.0,
            data.rider.right_hip.q[0]: np.deg2rad(30),
            data.rider.right_hip.u[0]: 0.0,
            data.rider.right_hip.q[1]: 0.0,
            data.rider.right_hip.u[1]: 0.0,
            data.rider.right_hip.q[2]: 0.0,
            data.rider.right_hip.u[2]: 0.0,
            data.rider.left_leg.q[0]: np.deg2rad(-45),
            data.rider.left_leg.u[0]: 0.0,
            data.rider.left_leg.q[1]: np.deg2rad(30),
            data.rider.left_leg.u[1]: 0.0,
            data.rider.right_leg.q[0]: np.deg2rad(-105),
            data.rider.right_leg.u[0]: 0.0,
            data.rider.right_leg.q[1]: np.deg2rad(-30),
            data.rider.right_leg.u[1]: 0.0,
            }

    print('initial guess : ', type(data.simulator.initial_conditions), data.simulator.initial_conditions)
    print("Number of variables in x:", len(data.x))
    print("Number of initial conditions provided:", len(data.simulator.initial_conditions))
    print()
    print(len(data.input_vars))

    s = data.metadata.straight_length
    d_lat = data.metadata.lateral_displacement
    path_length = 5 * s
    path_len = np.linspace(0, path_length, data.metadata.num_nodes)
    d_long1 = 3 * s
    d_long2 = 5 * s
    q1_path = []
    q2_path = []

    for q1 in path_len:
        if q1 < s:
            q2 = 0
        elif q1 <= d_long1 - s:
            progress = (q1 - s) / (d_long1 - 2 * s)
            q2 = d_lat * (1 - np.cos(np.pi * progress)) / 2
        elif q1 <= d_long1:
            q2 = d_lat
        elif q1 <= d_long2 - s:
            progress = (q1 - d_long1) / (d_long1 - 2 * s)
            q2 = d_lat * (1 + np.cos(np.pi * progress)) / 2
        else:
            q2 = 0
        q1_path.append(q1)
        q2_path.append(q2)
    q1_path = np.array(q1_path)
    q2_path = np.array(q2_path)

    # The following if/elif statements determine how the initial_guess matrix for the optimizer are created.
    if data.metadata.init_guess == data.metadata.init_guess.ZEROS:   # creates an initial guess matrix full of zeros
        print("Initial guess matrix consists of zeros.")
        t_arr = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)
        x_arr = np.zeros((len(data.simulator.initial_conditions), data.metadata.num_nodes))
    elif data.metadata.init_guess == data.metadata.init_guess.ONES:   # creates an initial guess matrix full of ones
        print("Initial guess matrix consists of ones.")
        t_arr = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)
        x_arr = np.ones((len(data.simulator.initial_conditions), data.metadata.num_nodes))
    elif data.metadata.init_guess == data.metadata.init_guess.INITIAL:   # creates an initial guess matrix of te initial guess values extrapolated over its length
        print("Initial guess matrix consists of init guess values.")
        t_arr = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)
        x_arr = np.zeros((len(data.simulator.initial_conditions), data.metadata.num_nodes))
        for i, key in enumerate(data.simulator.initial_conditions.keys()):
            x_arr[i, :] = data.simulator.initial_conditions[key]
    elif data.metadata.init_guess == data.metadata.init_guess.RANDOM:   # creates an initial guess matrix of random values between [-pi/2 and pi/2]
        print("Initial guess matrix consists of random values between -pi/2 and pi/2.")
        t_arr = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)
        x_arr = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(len(data.simulator.initial_conditions), data.metadata.num_nodes))
    elif data.metadata.init_guess == data.metadata.init_guess.PATH:
        t_arr = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)
        x_arr = np.zeros((len(data.simulator.initial_conditions), data.metadata.num_nodes))
        for i, key in enumerate(data.simulator.initial_conditions.keys()):
            x_arr[i, :] = data.simulator.initial_conditions[key]
        x_arr[0, :] = q1_path
        x_arr[1, :] = q2_path
    elif data.metadata.init_guess == data.metadata.init_guess.SIMULATED:   # creates an initial guess matrix by simulating the system (should be standard)
        print("Initial guess matrix consists of simulated initial guess.")
        t_arr, x_arr = data.simulator.solve(
            np.linspace(0, data.metadata.duration, data.metadata.num_nodes), "dae",
            rtol=1e-5, atol=1e-7)  # rtol=1e-5, atol=1e-7
        print('DAE solver: t_arr[-1]:', t_arr[-1], '&  duration:', data.metadata.duration)
        if t_arr[-1] != data.metadata.duration:
            print("DAE integration failed, integrating with solve_ivp.")
            t_arr, x_arr = data.simulator.solve(
                (0, data.metadata.duration), "solve_ivp",
                t_eval=np.linspace(0, data.metadata.duration, data.metadata.num_nodes))
    elif data.metadata.init_guess == data.metadata.init_guess.PREVIOUS:  # uses results of a previous simulation as an initial guess.
        OUTPUT_DIR ="output_PerturbedCycling"
        # make sure the previous result is of the same duration as your current sim, otherwise the x0 data will not match.
        result_dir = "result0"
        file_path = os.path.join(OUTPUT_DIR, result_dir, "solution_data.pkl")
        with open(file_path, "rb") as file:
            x_arr = cp.load(file)  # Load the cloudpickle file
            x_arr = x_arr[1:-5, :] # exclude the time, torque(s), energy, tracking and speed data, adjust for amounts of torques (1 torque=-4, 3 torques=-6)
        t_arr = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)

    print('lenghts of x_arr & t_arr -> ', len(x_arr), len(t_arr), x_arr.shape)
    print('ivp solver: t_arr[-1]:', t_arr[-1], '&  duration:', data.metadata.duration)

    data.initial_guess = np.concatenate(
        (x_arr.ravel(), np.zeros((len(data.input_vars)) * data.metadata.num_nodes)))