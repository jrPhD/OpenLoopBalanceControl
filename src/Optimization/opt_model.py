import symbrim as bm
import numpy as np
import sympy as sm
from src.Optimization.opt_container import SteerWith, DataStorage, SeatType, TorsoType
import bicycleparameters as bp
import matplotlib.pyplot as plt
from bicycleparameters import plot_eigenvalues
import sympy.physics.mechanics as me
from sympy.physics.vector import ReferenceFrame
from src.Optimization.opt_utils import get_all_symbols_from_model
"""
from src import (
    SimBicycleRider,
    SimRider,
    NeckPinJoint, NeckPinTorque, NeckPinSpringDamper,
    SimTorso, SimHead, SimPelvis, SimInterSeat,
    PinTorsoJoint, PinTorsoJointTorque, PinTorsoJointSpringDamper,
    SphericalTorsoJoint, SphericalTorsoJointSpringDamper, SphericalTorsoJointTorque,
    ShiftingSideLeanSeat, ShiftingSideLeanSeatTorque, ShiftingSideLeanSeatSpringDamper,
    InterSeatJoint
)
"""
from src.sim_bicycle_rider import SimBicycleRider
from src.simrider import SimRider
from src.SimNeckJoints import NeckPinJoint, NeckPinTorque, NeckPinSpringDamper, FixedNeck
from src.Sim_bodies import SimTorso, SimHead, SimPelvis, SimInterSeat
from src.SimTorsoJoints import (PinTorsoJoint, PinTorsoJointTorque, PinTorsoJointSpringDamper, FixedTorsoJoint,
                            SphericalTorsoJoint, SphericalTorsoJointSpringDamper, SphericalTorsoJointTorque)
from src.sim_seats import (ShiftingSideLeanSeat, ShiftingSideLeanSeatTorque,
                           ShiftingSideLeanSeatSpringDamper, InterSeatJoint)
from src.WhippleBicycle_Sprung_Steering import WhippleBicycleSprungSteering
from scipy.optimize import fsolve
from src.Optimization.opt_simulator import Simulator
from symbrim.brim import SideLeanSeat, SideLeanSeatTorque, SideLeanSeatSpringDamper, FixedSeat
from symbrim.bicycle import WhippleBicycleMoore, RigidRearFrameMoore, WhippleBicycle
from symbrim.rider import SphericalLeftHip, SphericalRightHip, SphericalHipTorque, SphericalHipSpringDamper
from symbrim.rider import TwoPinStickLeftLeg, TwoPinStickRightLeg, TwoPinLegTorque, TwoPinLegSpringDamper
from symbrim.brim import HolonomicPedals, SpringDamperPedals
from symbrim.bicycle import MasslessCranks, RigidRearFrame

def set_model(data: DataStorage):
    input_vars = sm.ImmutableMatrix()  # Create a matrix for the input variables
    # Set up the Bicycle
    if data.metadata.sprung_steering:
        bicycle = WhippleBicycleSprungSteering("bicycle")
    else:
        bicycle = WhippleBicycleMoore("bicycle")
    bicycle.front_frame = bm.RigidFrontFrame("front_frame")
    bicycle.rear_frame = RigidRearFrameMoore("rear_frame")
    #bicycle.rear_frame = RigidRearFrame("rear_frame")
    bicycle.front_wheel = bm.KnifeEdgeWheel("front_wheel")
    bicycle.rear_wheel = bm.KnifeEdgeWheel("rear_wheel")
    bicycle.front_tire = bm.NonHolonomicTire("front_tire")
    bicycle.rear_tire = bm.NonHolonomicTire("rear_tire")
    if data.metadata.model_legs == True:
        bicycle.cranks = MasslessCranks("cranks")
    bicycle.ground = bm.FlatGround("ground")
    # Set up the rider
    bicycle_rider = SimBicycleRider("bicycle_rider")
    #bicycle_rider = bm.BicycleRider("bicycle_rider")
    bicycle_rider.bicycle = bicycle

    # Set up the Rider
    if data.metadata.model_upper_body == True:
        #rider = bm.Rider("rider")
        rider = SimRider("rider")

        bicycle_rider.rider = rider
        rider.pelvis = SimPelvis("pelvis")
        if data.metadata.seat_type == data.metadata.seat_type.SIDELEAN:
            print('just the sidelean seat here')
            bicycle_rider.seat = SideLeanSeat("seat")
            slsd = SideLeanSeatSpringDamper("side_lean_spring_damper")
            bicycle_rider.seat.add_load_groups(slsd)
        elif data.metadata.seat_type == data.metadata.seat_type.SHIFTINGSIDELEAN:
            print('a whole shifting sidelean seat here')
            bicycle_rider.seat = InterSeatJoint("seat")
            rider.interseat = SimInterSeat("interseat")
            rider.shiftingsideleanseat = ShiftingSideLeanSeat("shifting_side_lean_seat")
            slsd = ShiftingSideLeanSeatSpringDamper("shifting_side_lean_spring_damper")
            rider.shiftingsideleanseat.add_load_groups(slsd)
        elif data.metadata.seat_type == data.metadata.seat_type.FIXED:
            print('a fixed seat for this model')
            bicycle_rider.seat = FixedSeat("seat")
        if data.metadata.model_torso == True:
            rider.torso = SimTorso("torso")
            if data.metadata.model == data.metadata.model.SINGLE_PENDULUM:
                if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
                    rider.torsojoint = FixedTorsoJoint("torsojoint")
                elif data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
                    print('single pendulum model at the torso joint, torque will now be added')
                    if data.metadata.torso_type == data.metadata.torso_type.PIN:
                        rider.torsojoint = PinTorsoJoint("torsojoint")
                    elif data.metadata.torso_type == data.metadata.torso_type.SPHERICAL:
                        rider.torsojoint = SphericalTorsoJoint("torsojoint")
                    print("torsojoint is:", type(rider.torsojoint))
                    if type(rider.torsojoint) == PinTorsoJoint:
                        torsd = PinTorsoJointSpringDamper("torso_spring_damper")
                        rider.torsojoint.add_load_groups(torsd)
                    if type(rider.torsojoint) == SphericalTorsoJoint:
                        torsd_sph = SphericalTorsoJointSpringDamper("torso_spherical_spring_damper")
                        rider.torsojoint.add_load_groups(torsd_sph)
            else:  ### if it's a double or triple pendulum
                rider.torsojoint = PinTorsoJoint("torsojoint")
                if type(rider.torsojoint) == PinTorsoJoint:
                    torsd = PinTorsoJointSpringDamper("torso_spring_damper")
                    rider.torsojoint.add_load_groups(torsd)
                if type(rider.torsojoint) == SphericalTorsoJoint:
                    torsd_sph = SphericalTorsoJointSpringDamper("torso_spherical_spring_damper")
                    rider.torsojoint.add_load_groups(torsd_sph)
        if data.metadata.model_head == True:
            rider.head = SimHead("head")
            if data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
                rider.neck = NeckPinJoint("neck")
                nesd = NeckPinSpringDamper("neck_spring_damper")
                rider.neck.add_load_groups(nesd)
            else:
                rider.neck = FixedNeck("neck")
            print('er zou een koppie op moeten zitten (met wellicht een springdamper)')
        if data.metadata.model_legs == True:
            rider.left_leg = TwoPinStickLeftLeg("left_leg")
            rider.right_leg = TwoPinStickRightLeg("right_leg")
            rider.left_hip = SphericalLeftHip("left_hip")
            rider.right_hip = SphericalRightHip("right_hip")
            bicycle_rider.pedals = HolonomicPedals("pedals")
            llegsd = TwoPinLegSpringDamper("left_leg_springdamper")
            rlegsd = TwoPinLegSpringDamper("right_leg_springdamper")
            rider.left_leg.add_load_groups(llegsd)
            rider.right_leg.add_load_groups(rlegsd)
            lhipsd = SphericalHipSpringDamper("left_hip_springdamper")
            rhipsd = SphericalHipSpringDamper("right_hip_springdamper")
            rider.left_hip.add_load_groups(lhipsd)
            rider.right_hip.add_load_groups(rhipsd)

        ### adding rider torques depending on the steerin inputs
        if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
            if type(bicycle_rider.seat) == SideLeanSeat:
                print('SideLean Seat torque activated')
                slst = SideLeanSeatTorque("side_lean_seat_torque")
                bicycle_rider.seat.add_load_groups(slst)
            elif type(bicycle_rider.seat) == InterSeatJoint:
                print('Shifting SideLean Seat torque activated')
                slst = ShiftingSideLeanSeatTorque("side_lean_seat_torque")
                rider.shiftingsideleanseat.add_load_groups(slst)
        if data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
            print('TorsoJoint torque activated')
            if type(rider.torsojoint) == PinTorsoJoint:
                print('its a pin torsojoint!')
                tort = PinTorsoJointTorque("torso_pin_torque")
                rider.torsojoint.add_load_groups(tort)
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                print('its a spherical torsojoint!')
                tort_sph = SphericalTorsoJointTorque("torso_spherical_torque")
                rider.torsojoint.add_load_groups(tort_sph)
        if data.metadata.steer_with == data.metadata.steer_with.SEAT_AND_TORSO_TORQUE:
            if type(bicycle_rider.seat) == SideLeanSeat:
                print('SideLean Seat torque activated')
                slst = SideLeanSeatTorque("side_lean_seat_torque")
                bicycle_rider.seat.add_load_groups(slst)
            elif type(bicycle_rider.seat) == InterSeatJoint:
                print('Shifting SideLean Seat torque activated')
                slst = ShiftingSideLeanSeatTorque("side_lean_seat_torque")
                rider.shiftingsideleanseat.add_load_groups(slst)
            if type(rider.torsojoint) == PinTorsoJoint:
                print('its a pin torsojoint!')
                tort = PinTorsoJointTorque("torso_pin_torque")
                rider.torsojoint.add_load_groups(tort)
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                print('its a spherical torsojoint!')
                tort_sph = SphericalTorsoJointTorque("torso_spherical_torque")
                rider.torsojoint.add_load_groups(tort_sph)
        if data.metadata.steer_with == data.metadata.steer_with.UPPER_BODY_TORQUE:
            if type(bicycle_rider.seat) == SideLeanSeat:
                print('SideLean Seat torque activated')
                slst = SideLeanSeatTorque("side_lean_seat_torque")
                bicycle_rider.seat.add_load_groups(slst)
            elif type(bicycle_rider.seat) == InterSeatJoint:
                print('Shifting SideLean Seat torque activated')
                slst = ShiftingSideLeanSeatTorque("side_lean_seat_torque")
                rider.shiftingsideleanseat.add_load_groups(slst)
            if type(rider.torsojoint) == PinTorsoJoint:
                print('its a pin torsojoint!')
                tort = PinTorsoJointTorque("torso_pin_torque")
                rider.torsojoint.add_load_groups(tort)
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                print('its a spherical torsojoint!')
                tort_sph = SphericalTorsoJointTorque("torso_spherical_torque")
                rider.torsojoint.add_load_groups(tort_sph)
            nect = NeckPinTorque("neck_torque")
            rider.neck.add_load_groups(nect)
        if data.metadata.steer_with == data.metadata.steer_with.LEG_TORQUE:
                print('lets get these legs workinn')
                llegt = TwoPinLegTorque("left_leg_torque")
                lhipt = SphericalHipTorque("left_hip_torque")
                rider.left_leg.add_load_groups(llegt)
                rider.left_hip.add_load_groups(lhipt)
                rlegt = TwoPinLegTorque("right_leg_torque")
                rhipt = SphericalHipTorque("right_hip_torque")
                rider.right_leg.add_load_groups(rlegt)
                rider.right_hip.add_load_groups(rhipt)

    # Define the model.
    bicycle_rider.define_connections()
    bicycle_rider.define_objects()
    print('BR submodels: ', bicycle_rider.submodels, ',   connections: ', bicycle_rider.connections)
    print('rider submodels:', rider.submodels, ',   connections: ', rider.connections)
    print('bicycle submodels:', bicycle.submodels, ', connections:', bicycle.connections)
    print(bicycle_rider.load_groups)
    ### In case multiple torques are combined to steer with, set ... i'll come back to this



    ### in case the damping/stiffness are constant, define them as a regular symbol ###
    if data.metadata.sprung_steering:
        fixed_stiffness = True
        data.fixed_stiffness = fixed_stiffness
        if fixed_stiffness == True:
            bicycle.symbols["k"] = sm.Symbol("k")
            bicycle.symbols["c"] = sm.Symbol("c")
            bicycle.symbols["q_ref"] = sm.Symbol("q_ref")
            print('adding symbols of steering spring')
        elif fixed_stiffness == False:
            bicycle.symbols["k"] = me.dynamicsymbols("k_steer")
            bicycle.symbols["c"] = sm.Symbol("c")
            bicycle.symbols["q_ref"] = sm.Symbol("q_ref")
            print('adding dynamic symbols of steering spring')

    if data.metadata.model_upper_body == True:
        if type(bicycle_rider.seat) == SideLeanSeat:
            slsd.symbols["k"] = sm.Symbol("k")
            slsd.symbols["c"] = sm.Symbol("c")
            slsd.symbols["q_ref"] = sm.Symbol("q_ref")
        elif type(bicycle_rider.seat) == InterSeatJoint:#ShiftingSideLeanSeat:
            slsd.symbols["k"] = sm.Symbol("k")
            slsd.symbols["c"] = sm.Symbol("c")
            slsd.symbols["q_ref"] = sm.Symbol("q_ref")
            bicycle_rider.seat.symbols["translation_factor"] = sm.Symbol("translation_factor")
            print('defining the translation factor')
        elif type(bicycle_rider.seat) == FixedSeat:
            None
    if data.metadata.model_torso == True:
        if type(rider.torsojoint) == PinTorsoJoint:
            torsd.symbols["k"] = sm.Symbol("k")
            torsd.symbols["c"] = sm.Symbol("c")
            torsd.symbols["q_ref"] = sm.Symbol("q_ref")
            print(torsd.symbols)
        if type(rider.torsojoint) == SphericalTorsoJoint:
            torsd_sph.symbols["k_flexion"] = sm.Symbol('k_flexion')
            torsd_sph.symbols["c_flexion"] = sm.Symbol('c_flexion')
            torsd_sph.symbols["q_ref_flexion"] = sm.Symbol('q_ref_flexion')
            torsd_sph.symbols["k_adduction"] = sm.Symbol('k_adduction')
            torsd_sph.symbols["c_adduction"] = sm.Symbol('c_adduction')
            torsd_sph.symbols["q_ref_adduction"] = sm.Symbol('q_ref_adduction')
            torsd_sph.symbols["k_rotation"] = sm.Symbol('k_rotation')
            torsd_sph.symbols["c_rotation"] = sm.Symbol('c_rotation')
            torsd_sph.symbols["q_ref_rotation"] = sm.Symbol('q_ref_rotation')
            print('torsd symbols: ', torsd_sph.symbols)
        else:
            None
    if data.metadata.model_head == True:
        if type(rider.neck) == NeckPinJoint:
            nesd.symbols["k"] = sm.Symbol('k')
            nesd.symbols["c"] = sm.Symbol('c')
            nesd.symbols["q_ref"] = sm.Symbol('q_ref')
        else:
            None
    if data.metadata.model_legs == True:
        bicycle.symbols["gear_ratio"] = sm.Symbol("gear_ratio")

        llegsd.symbols["k_knee"] = sm.Symbol("k_knee")
        llegsd.symbols["c_knee"] = sm.Symbol("c_knee")
        llegsd.symbols["q_ref_knee"] = sm.Symbol("q_ref_knee")
        llegsd.symbols["k_ankle"] = sm.Symbol("k_ankle")
        llegsd.symbols["c_ankle"] = sm.Symbol("c_ankle")
        llegsd.symbols["q_ref_ankle"] = sm.Symbol("q_ref_ankle")

        rlegsd.symbols["k_knee"] = sm.Symbol("k_knee")
        rlegsd.symbols["c_knee"] = sm.Symbol("c_knee")
        rlegsd.symbols["q_ref_knee"] = sm.Symbol("q_ref_knee")
        rlegsd.symbols["k_ankle"] = sm.Symbol("k_ankle")
        rlegsd.symbols["c_ankle"] = sm.Symbol("c_ankle")
        rlegsd.symbols["q_ref_ankle"] = sm.Symbol("q_ref_ankle")

        lhipsd.symbols["k_flexion"] = sm.Symbol('k_flexion')
        lhipsd.symbols["c_flexion"] = sm.Symbol('c_flexion')
        lhipsd.symbols["q_ref_flexion"] = sm.Symbol('q_ref_flexion')
        lhipsd.symbols["k_adduction"] = sm.Symbol('k_adduction')
        lhipsd.symbols["c_adduction"] = sm.Symbol('c_adduction')
        lhipsd.symbols["q_ref_adduction"] = sm.Symbol('q_ref_adduction')
        lhipsd.symbols["k_rotation"] = sm.Symbol('k_rotation')
        lhipsd.symbols["c_rotation"] = sm.Symbol('c_rotation')
        lhipsd.symbols["q_ref_rotation"] = sm.Symbol('q_ref_rotation')

        rhipsd.symbols["k_flexion"] = sm.Symbol('k_flexion')
        rhipsd.symbols["c_flexion"] = sm.Symbol('c_flexion')
        rhipsd.symbols["q_ref_flexion"] = sm.Symbol('q_ref_flexion')
        rhipsd.symbols["k_adduction"] = sm.Symbol('k_adduction')
        rhipsd.symbols["c_adduction"] = sm.Symbol('c_adduction')
        rhipsd.symbols["q_ref_adduction"] = sm.Symbol('q_ref_adduction')
        rhipsd.symbols["k_rotation"] = sm.Symbol('k_rotation')
        rhipsd.symbols["c_rotation"] = sm.Symbol('c_rotation')
        rhipsd.symbols["q_ref_rotation"] = sm.Symbol('q_ref_rotation')

    # Overwrite the symbols to match the paper.
    bicycle.q[:, 0] = me.dynamicsymbols("q1:9")
    bicycle.u[:, 0] = me.dynamicsymbols("u1:9")
    if data.metadata.model_upper_body:
        alpha = sm.Symbol("alpha")
        theta = sm.Symbol("theta")
        beta = sm.Symbol("beta")
        if type(bicycle_rider.seat) == FixedSeat:
            None
        else:
            int_frame_seat = me.ReferenceFrame("int_frame_seat")
            int_frame_seat.orient_axis(bicycle.rear_frame.saddle.frame, alpha,
                                       bicycle.rear_frame.wheel_hub.axis)
            bicycle_rider.seat.rear_interframe = int_frame_seat
        if data.metadata.model_torso:
            if type(rider.torsojoint) == PinTorsoJoint:
                int_frame_tor = me.ReferenceFrame("int_frame_torso")
                int_frame_tor.orient_axis(rider.pelvis.frame, theta,
                                        rider.pelvis.frame.y)
                rider.pelvis.tor_interframe = int_frame_tor
            elif type(rider.torsojoint) == FixedTorsoJoint:
                None
        if data.metadata.model_head:
            if type(rider.neck) == NeckPinJoint:
                int_frame_neck = me.ReferenceFrame("int_frame_neck")
                int_frame_neck.orient_axis(rider.torso.frame, beta,
                                        rider.torso.frame.y)
                rider.neck.neck_interframe = int_frame_neck
            elif type(rider.neck) == FixedNeck:
                None
        print('defined some alpha, theta & beta stuff')

        if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
            slst.symbols["T"] = me.dynamicsymbols("T_sls")
        elif data.metadata.steer_with == data.metadata.steer_with.SEAT_AND_TORSO_TORQUE:
            slst.symbols["T"] = me.dynamicsymbols("T_sls")
            if type(rider.torsojoint) == PinTorsoJoint:
                tort.symbols["T"] = me.dynamicsymbols("T_tor")
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                tort_sph.symbols["T_flexion"] = me.dynamicsymbols("T_tor_flexion")
                tort_sph.symbols["T_adduction"] = me.dynamicsymbols("T_tor_adduction")
                tort_sph.symbols["T_rotation"] = me.dynamicsymbols("T_tor_rotation")
        elif data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
            if type(rider.torsojoint) == PinTorsoJoint:
                tort.symbols["T"] = me.dynamicsymbols("T_tor")
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                tort_sph.symbols["T_flexion"] = me.dynamicsymbols("T_tor_flexion")
                tort_sph.symbols["T_adduction"] = me.dynamicsymbols("T_tor_adduction")
                tort_sph.symbols["T_rotation"] = me.dynamicsymbols("T_tor_rotation")
        elif data.metadata.steer_with == data.metadata.steer_with.UPPER_BODY_TORQUE:
            if type(bicycle_rider.seat) == SideLeanSeat:
                slst.symbols["T"] = me.dynamicsymbols("T_sls")
                print('defined the lean seat torque')
            if type(bicycle_rider.seat) == InterSeatJoint:
                slst.symbols["T"] = me.dynamicsymbols("T_sls")
                print('defined the shifting lean seat torque')
            if type(rider.torsojoint) == PinTorsoJoint:
                tort.symbols["T"] = me.dynamicsymbols("T_tor")
            if type(rider.torsojoint) == SphericalTorsoJoint:
                tort_sph.symbols["T_flexion"] = me.dynamicsymbols("T_tor_flexion")
                tort_sph.symbols["T_adduction"] = me.dynamicsymbols("T_tor_adduction")
                tort_sph.symbols["T_rotation"] = me.dynamicsymbols("T_tor_rotation")
            nect.symbols["T"] = me.dynamicsymbols("T_neck")
        elif data.metadata.steer_with == data.metadata.steer_with.LEG_TORQUE:
            llegt.symbols["T_knee"] = me.dynamicsymbols("T_lleg_knee")
            llegt.symbols["T_ankle"] = me.dynamicsymbols("T_lleg_ankle")
            rlegt.symbols["T_knee"] = me.dynamicsymbols("T_rleg_knee")
            rlegt.symbols["T_ankle"] = me.dynamicsymbols("T_rleg_ankle")

            lhipt.symbols["T_flexion"] = me.dynamicsymbols("T_lhip_flexion")
            lhipt.symbols["T_adduction"] = me.dynamicsymbols("T_lhip_adduction")
            lhipt.symbols["T_rotation"] = me.dynamicsymbols("T_lhip_rotation")
            rhipt.symbols["T_flexion"] = me.dynamicsymbols("T_rhip_flexion")
            rhipt.symbols["T_adduction"] = me.dynamicsymbols("T_rhip_adduction")
            rhipt.symbols["T_rotation"] = me.dynamicsymbols("T_rhip_rotation")


    if data.metadata.model_legs == True:
        bicycle.cranks.symbols["radius"]: 0.20
        bicycle.cranks.symbols["offset"]: 0.16
        bicycle.symbols["gear_ratio"]: 2


    bicycle_rider.define_kinematics()
    bicycle_rider.define_loads()
    bicycle_rider.define_constraints()
    print('load groups: ', bicycle_rider.load_groups)

    # Export model to a system object.
    system = bicycle_rider.to_system()
    speeds = np.linspace(0, 10, num=100)

    # Apply additional forces and torques to the system.
    g = sm.Symbol("g")
    system.apply_uniform_gravity(-g * bicycle.ground.get_normal(bicycle.ground.origin))

    #JR

    if data.metadata.pedaling_torque:
        print('Padaling torque included')
        ped_torque = me.dynamicsymbols("T_ped")
        system.add_actuators(me.TorqueActuator(
            ped_torque,     bicycle.rear_wheel.rotation_axis,
            bicycle.rear_wheel.frame, bicycle.rear_frame.wheel_hub.frame))
        input_vars = input_vars.col_join(sm.Matrix([ped_torque]))

    # constant torque friction (Coulomb), only valid for bicycle going
    # forwards!
    # from schwalbe marathon plus on bicyclerollingresistance.com
    rolling_resistance_torque = 2.2  # Nm
    system.add_actuators(me.TorqueActuator(
        -rolling_resistance_torque,  # opposite sign of pedaling torque
        bicycle.rear_wheel.rotation_axis,
        bicycle.rear_wheel.frame,
        bicycle.rear_frame.wheel_hub.frame))

    if data.metadata.extra_roll_torque:
        print('extra seat_torque included')
        extra_roll_torque = me.dynamicsymbols("T_ext_roll")
        system.add_actuators(me.TorqueActuator(extra_roll_torque,
            bicycle_rider.rider.pelvis.x,  # bicycle_rider.rider.pelvis.x,
            bicycle.rear_frame.saddle.frame, bicycle.ground.frame))
        input_vars = input_vars.col_join(sm.Matrix([extra_roll_torque]))





    if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
        print('adding seat torques to the system')
        seat_torque = me.dynamicsymbols("T_sls")
        if type(bicycle_rider.seat) == SideLeanSeat:
            system.add_actuators(me.TorqueActuator(
                seat_torque, bicycle_rider.seat.frame_lean_axis,  # bicycle_rider.rider.pelvis.x,
                bicycle_rider.rider.pelvis.frame, bicycle.rear_frame.saddle.frame))
        elif type(rider.shiftingsideleanseat) == ShiftingSideLeanSeat:
            system.add_actuators(me.TorqueActuator(
                seat_torque, rider.shiftingsideleanseat.frame_lean_axis,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.interseat.frame))
        input_vars = input_vars.col_join(sm.Matrix([slst.symbols["T"]]))
    if data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
        print('adding torsojoint torques to the system')
        if type(rider.torsojoint) == PinTorsoJoint:
            torso_torque = me.dynamicsymbols("T_tor")
            system.add_actuators(me.TorqueActuator(
                torso_torque, rider.torsojoint.pelvis.x,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.torso.frame))  # bicycle.rear_wheel.rotation_axis
            input_vars = input_vars.col_join(sm.Matrix([tort.symbols["T"]]))
        elif type(rider.torsojoint) == SphericalTorsoJoint:
            torso_flex_torque = me.dynamicsymbols("T_tor_flexion")
            torso_add_torque = me.dynamicsymbols("T_tor_adduction")
            torso_rot_torque = me.dynamicsymbols("T_tor_rotation")
            system.add_actuators(me.TorqueActuator(torso_flex_torque, rider.torsojoint.pelvis.y,
                                    rider.pelvis.frame, rider.torso.frame))
            system.add_actuators(me.TorqueActuator(torso_add_torque, rider.torsojoint.pelvis.x,
                                    rider.pelvis.frame, rider.torso.frame))
            system.add_actuators(me.TorqueActuator(torso_rot_torque, rider.torsojoint.pelvis.z,
                                    rider.pelvis.frame, rider.torso.frame))
            input_vars = input_vars.col_join(sm.Matrix([tort_sph.symbols["T_flexion"], tort_sph.symbols["T_adduction"], tort_sph.symbols["T_rotation"]]))
    if data.metadata.steer_with == SteerWith.SEAT_AND_TORSO_TORQUE:
        seat_torque = me.dynamicsymbols("T_sls")
        if type(bicycle_rider.seat) == SideLeanSeat:
            system.add_actuators(me.TorqueActuator(
                seat_torque, bicycle_rider.seat.frame_lean_axis,  # bicycle_rider.rider.pelvis.x,
                bicycle_rider.rider.pelvis.frame, bicycle.rear_frame.saddle.frame))
            torso_torque = me.dynamicsymbols("T_tor")
            system.add_actuators(me.TorqueActuator(
                torso_torque, rider.torsojoint.pelvis.x,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.torso.frame))  # bicycle.rear_wheel.rotation_axis
        if type(rider.shiftingsideleanseat) == ShiftingSideLeanSeat:
            system.add_actuators(me.TorqueActuator(
                seat_torque, rider.shiftingsideleanseat.frame_lean_axis,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.interseat.frame))
            torso_torque = me.dynamicsymbols("T_tor")
            system.add_actuators(me.TorqueActuator(
                torso_torque, rider.torsojoint.pelvis.x,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.torso.frame))  # bicycle.rear_wheel.rotation_axis
        input_vars = input_vars.col_join(sm.Matrix([slst.symbols["T"], tort.symbols["T"]]))
    if data.metadata.steer_with == SteerWith.LEG_TORQUE:
        print('adding leg torques to the system')
        llegknee_torque = me.dynamicsymbols("T_lleg_knee")
        system.add_actuators(me.TorqueActuator(llegknee_torque, rider.left_leg.thigh.y,
                                               rider.left_leg.shank.frame, rider.left_leg.thigh.frame))
        llegankle_torque = me.dynamicsymbols("T_lleg_ankle")
        system.add_actuators(me.TorqueActuator(llegankle_torque, rider.left_leg.shank.y,
                                               rider.left_leg.foot.frame, rider.left_leg.shank.frame))
        rlegknee_torque = me.dynamicsymbols("T_rleg_knee")
        system.add_actuators(me.TorqueActuator(rlegknee_torque, rider.right_leg.thigh.y,
                                               rider.right_leg.shank.frame, rider.right_leg.thigh.frame))
        rlegankle_torque = me.dynamicsymbols("T_rleg_ankle")
        system.add_actuators(me.TorqueActuator(rlegankle_torque, rider.right_leg.shank.y,
                                               rider.right_leg.foot.frame, rider.right_leg.shank.frame))

        lhip_flex_torque = me.dynamicsymbols("T_lhip_flexion")
        system.add_actuators(me.TorqueActuator(lhip_flex_torque, rider.pelvis.y,
                                               rider.pelvis.frame, rider.left_leg.thigh.frame))
        lhip_add_torque = me.dynamicsymbols("T_lhip_adduction")
        system.add_actuators(me.TorqueActuator(lhip_add_torque, rider.pelvis.x,
                                               rider.pelvis.frame, rider.left_leg.thigh.frame))
        lhip_rot_torque = me.dynamicsymbols("T_lhip_rotation")
        system.add_actuators(me.TorqueActuator(lhip_rot_torque, rider.pelvis.z,
                                               rider.pelvis.frame, rider.left_leg.thigh.frame))
        rhip_flex_torque = me.dynamicsymbols("T_rhip_flexion")
        system.add_actuators(me.TorqueActuator(rhip_flex_torque, rider.pelvis.y,
                                               rider.pelvis.frame, rider.right_leg.thigh.frame))
        rhip_add_torque = me.dynamicsymbols("T_rhip_adduction")
        system.add_actuators(me.TorqueActuator(rhip_add_torque, rider.pelvis.x,
                                               rider.pelvis.frame, rider.right_leg.thigh.frame))
        rhip_rot_torque = me.dynamicsymbols("T_rhip_rotation")
        system.add_actuators(me.TorqueActuator(rhip_rot_torque, rider.pelvis.z,
                                               rider.pelvis.frame, rider.right_leg.thigh.frame))
        input_vars = input_vars.col_join(
            sm.Matrix([lhipt.symbols["T_adduction"], lhipt.symbols["T_flexion"],  lhipt.symbols["T_rotation"],
                       llegt.symbols["T_knee"], llegt.symbols["T_ankle"],
                       rhipt.symbols["T_adduction"], rhipt.symbols["T_flexion"], rhipt.symbols["T_rotation"], #]))
                       rlegt.symbols["T_knee"], rlegt.symbols["T_ankle"]]))
    if data.metadata.steer_with == SteerWith.UPPER_BODY_TORQUE:
        print('adding upper body torques to the system')
        seat_torque = me.dynamicsymbols("T_sls")
        if type(bicycle_rider.seat) == SideLeanSeat:
            system.add_actuators(me.TorqueActuator(
                seat_torque, bicycle_rider.seat.frame_lean_axis,  # bicycle_rider.rider.pelvis.x,
                bicycle_rider.rider.pelvis.frame, bicycle.rear_frame.saddle.frame))
        if type(rider.shiftingsideleanseat) == ShiftingSideLeanSeat:
            system.add_actuators(me.TorqueActuator(
                seat_torque, rider.shiftingsideleanseat.frame_lean_axis,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.interseat.frame))
        input_vars = input_vars.col_join(sm.Matrix([slst.symbols["T"]]))
        if type(rider.torsojoint) == PinTorsoJoint:
            torso_torque = me.dynamicsymbols("T_tor")
            system.add_actuators(me.TorqueActuator(
                torso_torque, rider.torsojoint.pelvis.x,  # bicycle_rider.rider.pelvis.x,
                rider.pelvis.frame, rider.torso.frame))  # bicycle.rear_wheel.rotation_axis
            input_vars = input_vars.col_join(sm.Matrix([tort.symbols["T"]]))
        if type(rider.torsojoint) == SphericalTorsoJoint:
            torso_flex_torque = me.dynamicsymbols("T_tor_flexion")
            torso_add_torque = me.dynamicsymbols("T_tor_adduction")
            torso_rot_torque = me.dynamicsymbols("T_tor_rotation")
            system.add_actuators(me.TorqueActuator(torso_flex_torque, rider.torsojoint.pelvis.y,
                                    rider.pelvis.frame, rider.torso.frame))
            system.add_actuators(me.TorqueActuator(torso_add_torque, rider.torsojoint.pelvis.x,
                                    rider.pelvis.frame, rider.torso.frame))
            system.add_actuators(me.TorqueActuator(torso_rot_torque, rider.torsojoint.pelvis.z,
                                    rider.pelvis.frame, rider.torso.frame))
            input_vars = input_vars.col_join(sm.Matrix(
                [tort_sph.symbols["T_flexion"], tort_sph.symbols["T_adduction"], tort_sph.symbols["T_rotation"]]))
        if data.metadata.model_head:
            if type(rider.neck) == NeckPinJoint:
                neck_torque = me.dynamicsymbols("T_neck")
                system.add_actuators(me.TorqueActuator(neck_torque, rider.neck.torso.x,
                                        rider.torso.frame, rider.head.frame))
                input_vars = input_vars.col_join(sm.Matrix([nect.symbols["T"]]))
            elif type(rider.neck) == FixedNeck:
                None

    if data.metadata.sprung_steering == True and fixed_stiffness == False:
        input_vars = input_vars.col_join(sm.Matrix([bicycle.symbols["k"]]))

    print('all the input vars should beee:: ', input_vars, type(input_vars))

    # Specify the independent and dependent generalized coordinates and speeds.
    system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
    system.q_dep = [bicycle.q[4]]
    system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
    system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]
        ### adding the generalized speeds and coordinates of rider model:
    if data.metadata.model_upper_body:
        if type(bicycle_rider.seat) == FixedSeat:
            None
        elif type(bicycle_rider.seat) == SideLeanSeat:
            system.add_coordinates(*bicycle_rider.seat.q, independent=True)
            system.add_speeds(*bicycle_rider.seat.u, independent=True)
        elif type(bicycle_rider.seat) == InterSeatJoint:
            print(len(system.q_dep), len(system.q_ind), len(system.holonomic_constraints), len(system.velocity_constraints))
            print(len(system.u_dep), len(system.u_ind))
            system.add_coordinates(*rider.shiftingsideleanseat.q, independent=True)
            system.add_speeds(*rider.shiftingsideleanseat.u, independent=True)
            system.add_coordinates(*bicycle_rider.seat.q, independent=False)
            system.add_speeds(*bicycle_rider.seat.u, independent=False)
            system.add_holonomic_constraints(
                bicycle_rider.seat.q[0] - rider.shiftingsideleanseat.q[0] / bicycle_rider.seat.symbols["translation_factor"])
            system.velocity_constraints = system.velocity_constraints[:] + [
                system.holonomic_constraints[1].diff(me.dynamicsymbols._t)]
            print('velo cons:', len(system.velocity_constraints), system.velocity_constraints)
        if data.metadata.model_torso:
            if type(rider.torsojoint) == PinTorsoJoint:
                system.add_coordinates(*rider.torsojoint.q, independent=True)
                system.add_speeds(*rider.torsojoint.u, independent=True)
            elif type(rider.torsojoint) == SphericalTorsoJoint:
                system.add_coordinates(rider.torsojoint.q[0], independent=True)
                system.add_speeds(rider.torsojoint.u[0], independent=True)
                system.add_coordinates(rider.torsojoint.q[1], independent=True)
                system.add_speeds(rider.torsojoint.u[1], independent=True)
                system.add_coordinates(rider.torsojoint.q[2], independent=True)
                system.add_speeds(rider.torsojoint.u[2], independent=True)
            elif type(rider.torsojoint) == FixedTorsoJoint:
                None
        if data.metadata.model_head:
            if type(rider.neck) == NeckPinJoint:
                system.add_coordinates(*rider.neck.q, independent=True)
                system.add_speeds(*rider.neck.u, independent=True)
            elif type(rider.neck) == FixedNeck:
                None
        if data.metadata.model_legs:
            system.add_coordinates(rider.left_hip.q[0], independent=True)
            system.add_speeds(rider.left_hip.u[0], independent=True)
            system.add_coordinates(rider.left_hip.q[1], independent=True)
            system.add_speeds(rider.left_hip.u[1], independent=True)
            system.add_coordinates(rider.left_hip.q[2], independent=True)
            system.add_speeds(rider.left_hip.u[2], independent=True)
            system.add_coordinates(rider.right_hip.q[0], independent=True)
            system.add_speeds(rider.right_hip.u[0], independent=True)
            system.add_coordinates(rider.right_hip.q[1], independent=True)
            system.add_speeds(rider.right_hip.u[1], independent=True)
            system.add_coordinates(rider.right_hip.q[2], independent=True)
            system.add_speeds(rider.right_hip.u[2], independent=True)

            system.add_coordinates(rider.left_leg.q[0], independent=True)
            system.add_speeds(rider.left_leg.u[0], independent=True)
            system.add_coordinates(rider.left_leg.q[1], independent=True)
            system.add_speeds(rider.left_leg.u[1], independent=True)
            system.add_coordinates(rider.right_leg.q[0], independent=True)
            system.add_speeds(rider.right_leg.u[0], independent=True)
            system.add_coordinates(rider.right_leg.q[1], independent=True)
            system.add_speeds(rider.right_leg.u[1], independent=True)
    if data.metadata.task == data.metadata.task.PERTURBED_CYCLING:

        wind = me.dynamicsymbols("wind")
        system.add_loads(
            me.Force(bicycle.rear_frame.saddle.point,
                     wind * bicycle.rear_frame.wheel_hub.axis))

        data.wind = wind
        def force_amplitude(x):
            #return 5 * (9 - 15 * np.cos(x / 1) + 2 * np.sin(15 * x) + 7 * np.cos(4 * x))
            return (-20 / ((x-0.05) + 20 * 0.00917431)) + 100 + np.cos((x-0.05) / 5) + 4 * np.sin(3 * (x-0.05)) + 8 * np.cos(4 * (x-0.05))
        x_values = np.linspace(0, data.metadata.duration, data.metadata.num_nodes)
        print('length of wind array:', len(x_values))
        wind_array = force_amplitude(x_values)
        data.wind_array = wind_array
        {data.wind: data.wind_array}

    rider_stability = False
    if rider_stability == True:
        Y = ReferenceFrame('Y')
        Y.orient_axis(bicycle.ground.frame, bicycle.q[2], bicycle.ground.frame.z)

        Lhead = sm.Matrix(rider.head.frame.dcm(Y))
        Ltorso = sm.Matrix(rider.torso.frame.dcm(Y))
        Lpelvis = sm.Matrix(rider.pelvis.frame.dcm(Y))

        Lhead[2, 2] = me.dynamicsymbols("lean_head")
        Ltorso[2, 2] = me.dynamicsymbols("lean_torso")
        Lpelvis[2, 2] = me.dynamicsymbols("lean_pelvis")

        angles = sm.Matrix([Lhead[2,2], Ltorso[2,2], Lpelvis[2,2]])
        print(angles)

    # Simple check to see if the system is valid.
    system.validate_system()
    # Form the equations of motion. Note: LU solve may lead to zero divisions.
    essential_eoms = system.form_eoms(constraint_solver="CRAMER")
    eoms = system.kdes.col_join(essential_eoms).col_join(
        system.holonomic_constraints).col_join(system.nonholonomic_constraints)


    # Obtain constant parameters.
    bicycle_params = bp.Bicycle(
        data.metadata.bicycle_parametrization,
        pathToData=data.metadata.parameter_data_dir)
    # Rough esitmations of bicycle parameters for plotting purposes.
    if data.metadata.bicycle_parametrization == "Fisher":
        bicycle_params.parameters["Measured"]["hbb"] = 0.3
        bicycle_params.parameters["Measured"]["lcs"] = 0.44
        bicycle_params.parameters["Measured"]["lsp"] = 0.22
        bicycle_params.parameters["Measured"]["lst"] = 0.53
        bicycle_params.parameters["Measured"]["lamst"] = 1.29
        bicycle_params.parameters["Measured"]["whb"] = 0.6
        bicycle_params.parameters["Measured"]["LhbR"] = 1.11
        bicycle_params.parameters["Measured"]["LhbF"] = 0.65
    elif data.metadata.bicycle_parametrization == "Pista":
        bicycle_params.parameters["Measured"]["hbb"] = 0.27
        bicycle_params.parameters["Measured"]["lcs"] = 0.41
        bicycle_params.parameters["Measured"]["lsp"] = 0.24
        bicycle_params.parameters["Measured"]["lst"] = 0.52
        bicycle_params.parameters["Measured"]["lamst"] = 1.32
        bicycle_params.parameters["Measured"]["whb"] = 0.42
        bicycle_params.parameters["Measured"]["LhbR"] = 1.18
        bicycle_params.parameters["Measured"]["LhbF"] = 0.51
    if not data.metadata.bicycle_only:
        bicycle_params.add_rider(data.metadata.rider_parametrization, reCalc=True)
    constants = bicycle_rider.get_param_values(bicycle_params)
    constants[g] = 9.81
    if data.metadata.sprung_steering:
        print('adding spring stiffness to the sprung steering frame')
        if fixed_stiffness == True:
            constants.update({
                bicycle.symbols["k"]: 30.0,
                bicycle.symbols["c"]: 0.0,
                bicycle.symbols["q_ref"]: 0.0})
        elif fixed_stiffness == False:
            constants.update({
                bicycle.symbols["c"]: 0.0,
                bicycle.symbols["q_ref"]: 0.0})
    if data.metadata.sprung_steering:
        print("bicycle.symbols keys+values: ", bicycle.symbols.keys(),"=", list(constants.values())[-3:])

    if data.metadata.model_upper_body:
        if data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
            if type(bicycle_rider.seat) == FixedSeat:
                None
            else:
                constants.update({
                    slsd.symbols["k"]: 0,
                    slsd.symbols["c"]: 0,
                    slsd.symbols["q_ref"]: 0.0})
        elif data.metadata.steer_with == data.metadata.steer_with.LEG_TORQUE:
            if type(bicycle_rider.seat) == FixedSeat:
                None
            else:
                constants.update({
                    slsd.symbols["k"]: 0,
                    slsd.symbols["c"]: 0,
                    slsd.symbols["q_ref"]: 0.0})
        else:       ## For all these cases, the seat joint would be controlled by an input torque
            constants.update({
                slsd.symbols["k"]: 0,
                slsd.symbols["c"]: 0,
                slsd.symbols["q_ref"]: 0.0})
        if type(bicycle_rider.seat) == SideLeanSeat:
            constants.update({bicycle_rider.seat.symbols["alpha"]: -0.2})
        elif type(bicycle_rider.seat) == InterSeatJoint:
            constants.update({bicycle_rider.seat.symbols["translation_factor"]: 6.98}) ## comes from 0.698rad/0.1m (= ~45deg/10cm)  7.854
            constants.update({rider.shiftingsideleanseat.symbols["alpha"]: -0.2})
            print('so the translation_factor should have been assigned a value')
        elif type(bicycle_rider.seat) == FixedSeat:
            constants.update({
                bicycle_rider.seat.symbols["yaw"]: 0.0,
                bicycle_rider.seat.symbols["pitch"]: 0.01,
                bicycle_rider.seat.symbols["roll"]: 0.0})

        if data.metadata.model_torso == True:
            if data.metadata.model == data.metadata.model.SINGLE_PENDULUM and data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
                print('single pendulum model at the seat joint, so torso yaw, pitch and roll angles are defined.')
                constants.update({
                    rider.torsojoint.symbols["yaw"]: 0.0,
                    rider.torsojoint.symbols["pitch"]: 0.01,
                    rider.torsojoint.symbols["roll"]: 0.0})
            else:
                if type(rider.torsojoint) == PinTorsoJoint:
                    constants.update({
                        rider.torsojoint.symbols["theta"]: 0.01,
                        torsd.symbols["k"]: 0,
                        torsd.symbols["c"]: 0,
                        torsd.symbols["q_ref"]: 0.0})
                if type(rider.torsojoint) == SphericalTorsoJoint:
                    constants.update({
                        torsd_sph.symbols["k_flexion"]: 5,
                        torsd_sph.symbols["c_flexion"]: 10,
                        torsd_sph.symbols["q_ref_flexion"]: 0.01,
                        torsd_sph.symbols["k_adduction"]: 0,
                        torsd_sph.symbols["c_adduction"]: 10,
                        torsd_sph.symbols["q_ref_adduction"]: 0.0,
                        torsd_sph.symbols["k_rotation"]: 8,
                        torsd_sph.symbols["c_rotation"]: 10,
                        torsd_sph.symbols["q_ref_rotation"]: 0.0})
            if data.metadata.model_head == True:
                if data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
                    constants.update({
                        rider.neck.symbols["beta"]: 0.01,
                        nesd.symbols["k"]: 0,
                        nesd.symbols["c"]: 12,
                        nesd.symbols["q_ref"]: np.deg2rad(0)})
                else:
                    constants.update({
                        rider.neck.symbols["yaw"]: 0.0,
                        rider.neck.symbols["pitch"]: 0.01,
                        rider.neck.symbols["roll"]: 0.0})
        if data.metadata.model_legs == True:
            constants.update({
                llegsd.symbols["k_knee"]: 1.0,
                llegsd.symbols["c_knee"]: 10,
                llegsd.symbols["q_ref_knee"]: 4.7,
                llegsd.symbols["k_ankle"]: 1.0,
                llegsd.symbols["c_ankle"]: 8,
                llegsd.symbols["q_ref_ankle"]: 1.57,
                rlegsd.symbols["k_knee"]: 1.0,
                rlegsd.symbols["c_knee"]: 10,
                rlegsd.symbols["q_ref_knee"]: 4.7,
                rlegsd.symbols["k_ankle"]: 1.0,
                rlegsd.symbols["c_ankle"]: 8.0,
                rlegsd.symbols["q_ref_ankle"]: 1.57,
                lhipsd.symbols["k_flexion"]: 10,
                lhipsd.symbols["c_flexion"]: 15,
                lhipsd.symbols["q_ref_flexion"]: 2.35,
                lhipsd.symbols["k_adduction"]: 1,
                lhipsd.symbols["c_adduction"]: 10,
                lhipsd.symbols["q_ref_adduction"]: 0.0,
                lhipsd.symbols["k_rotation"]: 3,
                lhipsd.symbols["c_rotation"]: 15,
                lhipsd.symbols["q_ref_rotation"]: 0.0,
                rhipsd.symbols["k_flexion"]: 10,
                rhipsd.symbols["c_flexion"]: 15,
                rhipsd.symbols["q_ref_flexion"]: 2.35,
                rhipsd.symbols["k_adduction"]: 1.0,
                rhipsd.symbols["c_adduction"]: 10,
                rhipsd.symbols["q_ref_adduction"]: 0.0,
                rhipsd.symbols["k_rotation"]: 3.0,
                rhipsd.symbols["c_rotation"]: 15,
                rhipsd.symbols["q_ref_rotation"]: 0.0,
                bicycle.cranks.symbols["radius"]: 0.20,
                bicycle.cranks.symbols["offset"]: 0.16,
                bicycle.symbols["gear_ratio"]: 2
            })

        approximate_alpha = -(bicycle_params.parameters["Benchmark"]["lam"] +
                              bicycle_params.human.CFG["somersault"]).nominal_value
        # The somersault angle is generally based on straight arms, therefore the value
        # should be a bit higher to account for the bent arms and steering.
        if data.metadata.bicycle_parametrization == "Browser":
            offset = 0.125
        else:
            offset = 0.1
        constants[alpha] = approximate_alpha - offset

    print('los constantes:', constants)
    syms = get_all_symbols_from_model(bicycle_rider)
    symsbicicleta = get_all_symbols_from_model(bicycle)
    print(bicycle.symbols.values())
    missing_constants = syms.difference(constants.keys()).difference({
        bicycle.symbols.get("gear_ratio", 0), 0, *input_vars})
    print('missing constants: ', missing_constants)
    print('System q: ', system.q, '. System u: ', system.u)

    if data.metadata.model_upper_body:
        missing_constants = missing_constants.difference(
            bicycle_rider.seat.symbols.values())
    if missing_constants:
        rear_constants_estimates = {
            bicycle.rear_frame.symbols["d4"]: 0.42,
            bicycle.rear_frame.symbols["d5"]: -0.55,
            bicycle.rear_frame.symbols["l_bbx"]: 0.40,
            bicycle.rear_frame.symbols["l_bbz"]: 0.22,
        }
        front_constants_estimates = {
            bicycle.front_frame.symbols["d6"]: -0.17,
            bicycle.front_frame.symbols["d7"]: 0.29,
            bicycle.front_frame.symbols["d8"]: -0.37,
        }

        if (data.metadata.model_upper_body and
                missing_constants.difference(rear_constants_estimates.keys())):
            raise ValueError(f"Missing constants: {missing_constants}")
        elif missing_constants.difference(rear_constants_estimates.keys()).difference(
                front_constants_estimates.keys()):
            raise ValueError(f"Missing constants: {missing_constants}")
        estimated_constants = {
            sym: rear_constants_estimates.get(sym, front_constants_estimates.get(sym))
            for sym in missing_constants
        }
        print(f"Estimated constants, which are used for visualization purposes only: "
              f"{estimated_constants}.")
        constants.update(estimated_constants)

    # Include missing rider mass into the rear frame
    rear_body = bicycle.rear_frame.body

    if data.metadata.model_legs == False:
        # Add the inertia of the legs to the rear frame
        print("adding the inertia of the legs to the rear frame")
        leg = bm.TwoPinStickLeftLeg("left_leg")
        q_hip = me.dynamicsymbols("q_hip")
        leg.define_all()

        leg.hip_interframe.orient_axis(bicycle.rear_frame.saddle.frame, q_hip,
                                       bicycle.rear_frame.wheel_hub.axis)
        offset = rider.pelvis.symbols["hip_width"] * bicycle.rear_frame.saddle.frame.y
        leg.hip_interpoint.set_pos(
            rider.pelvis.left_hip_point,
            rider.pelvis.right_hip_point.pos_from(rider.pelvis.left_hip_point) / 2)
        val_dict = {leg.q[1]: 0, **leg.get_param_values(bicycle_params), **constants}
        if type(bicycle_rider.seat) == SideLeanSeat:
            val_dict[bicycle_rider.seat.q[0]] = 0  # Replace with the actual value
            val_dict[bicycle_rider.seat.u[0]] = 0  # Replace with the actual value
        elif type(bicycle_rider.seat) == InterSeatJoint:
            val_dict[rider.shiftingsideleanseat.q[0]] = 0  # Replace with the actual value
            val_dict[bicycle_rider.seat.q[0]] = 0  # Replace with the actual value
            val_dict[rider.shiftingsideleanseat.u[0]] = 0  # Replace with the actual value
            val_dict[bicycle_rider.seat.u[0]] = 0  # Replace with the actual value

        v = leg.foot_interpoint.pos_from(bicycle.rear_frame.bottom_bracket).to_matrix(
            bicycle.rear_frame.wheel_hub.frame).xreplace(val_dict).simplify()
        val_dict[q_hip], val_dict[leg.q[0]] = fsolve(
            sm.lambdify([(q_hip, leg.q[0])], [v[0], v[2]]), (0.6, 1.5))
        additional_inertia = me.Dyadic(0)
        print('val_dict:', val_dict)
        additional_mass = sm.S.Zero
        for body in leg.system.bodies:
            additional_inertia += 2 * body.parallel_axis(rear_body.masscenter)
            additional_inertia += 2 * body.mass * (me.inertia(
                rear_body.frame, 1, 1, 1) * offset.dot(offset) - offset.outer(offset))
            additional_mass += 2 * body.mass
        extra_i_vals = sm.lambdify(
            val_dict.keys(), additional_inertia.to_matrix(rear_body.frame),
            cse=True)(*val_dict.values())
        i_rear = rear_body.central_inertia.to_matrix(rear_body.frame)
        constants[rear_body.mass] += float(additional_mass.xreplace(val_dict))

        for idx in [(0, 0), (1, 1), (2, 2), (2, 0)]:
            constants[i_rear[idx]] += float(extra_i_vals[idx])


    data.bicycle_rider = bicycle_rider
    data.bicycle = bicycle
    if data.metadata.model_upper_body:
        data.rider = rider
    data.system = system
    data.eoms = eoms

    # data.input_vars = sm.ImmutableMatrix(sorted(input_vars, key=lambda ri: ri.name))
    # #v_angles = sm.ImmutableMatrix(angles)
    # #print(type(input_vars), type(v_angles))
    # #data.angles = sm.ImmutableMatrix(sorted(v_angles, key=lambda ri: ri.name))
    # print('the data.input_vars thing is ->', data.input_vars)

    # print('------------')
    # print(data.metadata.part_personalized_constants.keys())

    # for key in data.metadata.part_personalized_constants.keys():
    #     try:
    # constants.update(data.metadata.part_personalized_constants)
        # except:
        #     print('Error',f'{key}','have not been over written')
    # constants['']

    # print(constants.keys())

    data.constants = constants

    #Express angular velocity of head frame

    head_body = rider.head.body

    ang_vel_mat_head = rider.head.frame.ang_vel_in(system.frame).to_matrix(rider.head.frame)
    acc_mat_head = head_body.masscenter.acc(system.frame).to_matrix(rider.head.frame)
    acc_mat_head = acc_mat_head - g*system.frame.z.to_matrix(rider.head.frame)

    mAx = acc_mat_head[0]
    mAy = acc_mat_head[1]
    mAz = acc_mat_head[2]

    mWx = ang_vel_mat_head[0]
    mWy = ang_vel_mat_head[1]
    mWz = ang_vel_mat_head[2]

    ax, ay, az = me.dynamicsymbols('a_x, a_y, a_z')
    omega_x, omega_y, omega_z = me.dynamicsymbols('omega_x, omega_y, omega_z')

    data.eoms = data.eoms.col_join(sm.Matrix([ax - mAx, ay - mAy, az - mAz]))
    data.eoms = data.eoms.col_join(sm.Matrix([omega_x - mWx, omega_y - mWy, omega_z - mWz]))

    for symb in [ax, ay, az, omega_x, omega_y, omega_z]:
        system.add_speeds(symb, independent=False)
    # input_vars = input_vars.col_join(sm.Matrix([ax, ay, az, omega_x, omega_y, omega_z]))

    # input_vars = input_vars + [ax, ay, az, omega_x, omega_y, omega_z]
    data.input_vars = sm.ImmutableMatrix(sorted(input_vars, key=lambda ri: ri.name))

    print('the data.input_vars are ->', data.input_vars)

    # return(ang_vel_mat_head, acc_mat_head)

    # print('------------')

    # print(data.constants)








def set_simulator(data: DataStorage) -> None:
    simulator = Simulator(data.system)

    # for key in data.metadata.part_personalized_constants.keys():
    #     try:
    #         data.constants[f'{key}'] = data.metadata.part_personalized_constants[f'{key}']
    #     except:
    #         print('Error',f'{key}','have not been over written')


    # print('ello')


    simulator.constants = data.constants
    simulator.inputs = {ri: lambda t, x: 0.0 for ri in data.input_vars}
    if data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        print('adding the wind function in the simulator')
        simulator.inputs[data.wind] = lambda t, x: (-20 / ((t-0.05) + 20 * 0.00917431)) + 100 + np.cos((t-0.05) / 5) + 4 * np.sin(3 * (t-0.05)) + 8 * np.cos(4 * (t-0.05))
        #simulator.inputs[data.wind] = lambda t, x: 0.0
        #simulator.inputs[data.wind] = lambda t, x: 5 * ((9 - 15 * np.cos(t / 1) + 2 * np.sin(15 * t) + 7 * np.cos(4 * t)))
    else:
        None
    simulator.initial_conditions = {xi: 0.0 for xi in data.x}
    #simulator.initial_conditions = {xi: 0.0 for xi in data.angles}
    simulator.initial_conditions[data.bicycle.q[4]] = 0.314 #Why ???
    simulator.initialize(False)
    data.simulator = simulator

