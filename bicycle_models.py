import pickle
import platform
import warnings
from IPython.display import display

import sympy as sm
import sympy.physics.mechanics as me

import bicycleparameters as bp
from symbrim.bicycle import RigidRearFrameMoore, WhippleBicycleMoore
import symbrim
import symbrim as sb


TIME_SYM = me.dynamicsymbols._t


def generate_bicycle_rider_model():
    """
    Returns
    =======
    system : sympy.physics.mechanics.system.System

    """

    bicycle = symbrim.WhippleBicycle('bicycle')
    bicycle.ground = symbrim.FlatGround("ground")

    bicycle.front_frame = symbrim.RigidFrontFrame("front_frame")
    bicycle.rear_frame = symbrim.RigidRearFrame.from_convention(
        "moore", "rear_frame")

    bicycle.front_wheel = symbrim.KnifeEdgeWheel("front_wheel")
    bicycle.rear_wheel = symbrim.KnifeEdgeWheel("rear_wheel")

    bicycle.front_tire = symbrim.NonHolonomicTire("front_tire")
    bicycle.rear_tire = symbrim.NonHolonomicTire("rear_tire")

    bicycle_rider = bicycle

    bicycle_rider.define_all()
    # NOTE : This system seems to be a copy of the bicycle.system, i.e. if you
    # mutate the returned system
    system = bicycle_rider.to_system()

    # TODO: bicycle.ground.origin is not used internally in get_normal(), so
    # not clear what it is for, see
    # https://github.com/mechmotum/symbrim/issues/158
    # It seems that `normal` points opposite of gravity (from ground to sky),
    # but that is not the Moore convention, which points downward.
    g = sm.symbols("g")
    normal = bicycle.ground.get_normal(bicycle.ground.origin)
    system.apply_uniform_gravity(-g*normal)

    # Torque applied between rear wheel and rear frame to propel the bicycle.
    # Positive torque pushes rear frame in positive motion about axis.
    # TODO : Which direction is the positive direction?
    pedaling_torque = me.dynamicsymbols("T6")
    system.add_actuators(
        me.TorqueActuator(
            pedaling_torque,
            bicycle.rear_frame.wheel_hub.axis,
            bicycle.rear_wheel.frame,
            # TODO : Why rear_frame.wheel_hub.frame and not just
            # rear_frame.frame?
            bicycle.rear_frame.wheel_hub.frame,
        )
    )

    # Torque applied between the front frame and rear frame about the steer
    # axis to steer the bicycle.
    steer_torque = me.dynamicsymbols("T7")
    system.add_actuators(
        me.TorqueActuator(
            steer_torque,
            bicycle.rear_frame.steer_hub.axis,
            bicycle.front_frame.steer_hub.frame,
            bicycle.rear_frame.steer_hub.frame,
        )
    )

    # Torque applied between the ground and the rear frame about the roll axis.
    roll_torque = me.dynamicsymbols("T4")
    yaw_frame = me.ReferenceFrame('yaw_frame')
    yaw_frame.orient_axis(bicycle.ground.frame, bicycle.ground.frame.z,
                          bicycle.q[2])
    system.add_actuators(
        me.TorqueActuator(
            roll_torque,
            yaw_frame.x,
            bicycle.rear_frame.wheel_hub.frame,
            bicycle.ground.frame,
        )
    )

    # Force applied to the saddle, expressed in the rear frame unit vectors.
    Fx, Fy, Fz = me.dynamicsymbols('F_x, F_y, F_z')
    system.add_loads(me.Force(
        bicycle.rear_frame.saddle.point,
        Fx*bicycle.rear_frame.saddle.frame.x +
        Fy*bicycle.rear_frame.saddle.frame.y +
        Fz*bicycle.rear_frame.saddle.frame.z
    ))

    # Before forming the EoMs we need to specify which generalized coordinates
    # and speeds are independent and which are dependent.
    # q indep : q1, q2, q3, q4, q6, q7, q8
    # u indep : u4, u6, u7
    system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
    system.q_dep = [bicycle.q[4]]
    system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
    system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]
    system.validate_system()
    system.form_eoms(constraint_solver="CRAMER")

    #specifieds = sm.Matrix([
        #roll_torque,
        #pedaling_torque,
        #steer_torque,
        #Fx,
        #Fy,
        #Fz,
    #])

    return bicycle_rider, system


def generate_model(model_name, config):
    t = me.dynamicsymbols._t
    bicycle = sb.WhippleBicycle(f"{model_name}")
    assert type(bicycle) is WhippleBicycleMoore
    bicycle.rear_frame = sb.RigidRearFrame.from_convention("moore", "rear_frame")
    assert type(bicycle.rear_frame) is RigidRearFrameMoore
    bicycle.rear_wheel = sb.KnifeEdgeWheel("rear_wheel")
    bicycle.rear_tire = sb.NonHolonomicTire("rear_tire")

    bicycle.ground = sb.FlatGround("ground")
    bicycle.front_frame = sb.RigidFrontFrame("front_frame")
    bicycle.front_wheel = sb.KnifeEdgeWheel("front_wheel")
    bicycle.front_tire = sb.NonHolonomicTire("front_tire")

    assert len(bicycle.submodels) == 5
    assert len(bicycle.connections) == 2

    bicycle.define_all()
    system = bicycle.to_system()

    normal = bicycle.ground.get_normal(bicycle.ground.origin)

    # Add loads and actuators

    # Gravity
    g = sm.symbols("g")
    system.apply_uniform_gravity(-g * normal)

    m_rider = sm.symbols('m_rider')
    system.add_loads(me.Force(bicycle.rear_frame.saddle.point, -g*m_rider*normal))

    pedaling_torque = me.dynamicsymbols("pedaling_torque")
    system.add_actuators(
        me.TorqueActuator(
            pedaling_torque,
            bicycle.rear_frame.wheel_hub.axis,
            bicycle.rear_wheel.frame,
            bicycle.rear_frame.wheel_hub.frame,
        )
    )


    r = sm.Matrix([pedaling_torque])



    if config['steer_torque'] == True :

        # Steer torque
        steer_torque = me.dynamicsymbols("steer_torque")
        system.add_actuators(
            me.TorqueActuator(
                steer_torque,
                bicycle.rear_frame.steer_hub.axis,
                bicycle.rear_frame.steer_hub.frame,
                bicycle.front_frame.steer_hub.frame,
            )
        )

        r = r.col_join(sm.Matrix([steer_torque]))

    if config['roll_control'] == True :


        roll_moment = me.dynamicsymbols("M_x")

        system.add_actuators(
            me.TorqueActuator(
                roll_moment,
                bicycle.rear_frame.body.frame.x,
                bicycle.ground.frame,
                bicycle.rear_frame.body.frame,
            )
        )



        rider_saddle_force_y = me.dynamicsymbols("F_y")
        rider_saddle_force_z = me.dynamicsymbols("F_z")
        system.add_loads(me.Force(bicycle.rear_frame.saddle.point, rider_saddle_force_y*bicycle.rear_frame.saddle.frame.y))
        system.add_loads(me.Force(bicycle.rear_frame.saddle.point, rider_saddle_force_z*bicycle.rear_frame.saddle.frame.z))


        r = r.col_join(sm.Matrix([roll_moment, rider_saddle_force_y, rider_saddle_force_z]))




    # Before forming the EoMs we need to specify which generalized coordinates
    # and speeds are independent and which are dependent.

    #q indep : q1, q2, q3, q4, q6, q7, q8
    #u indep : u4, u6, u7

    system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
    system.q_dep = [bicycle.q[4]]
    system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
    system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]
    system.validate_system()

    try:
        system.validate_system()
    except ValueError as e:
        print("\n\nERROR : ")
        display(e)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eoms = system.form_eoms(constraint_solver="CRAMER")

    # The equations of motions are generated as a
    #                                   "sympy.matrices.dense.MutableDenseMatrix"

    # %% Parametrization


    bike_params = bp.Bicycle("Browser", pathToData="data")
    # bike_params.add_rider("Jason", reCalc=True)

    constants = bicycle.get_param_values(bike_params)
    constants[g] = 9.81  # Don't forget to specify the gravitational constant.

    constants[m_rider] = 80

    # print("\n\nConstants of the model:")
    # print(constants)

    # missing_symbols = bicycle.get_all_symbols().difference(constants.keys())

    # print("\n\nIs there any missing constant? -->")
    # print(missing_symbols)


    # eoms = eoms.col_join(sm.Matrix([disturbance - f_disturb(t)]))

    x = system.q.col_join(system.u)
    # x = x.col_join(sm.Matrix([disturbance]))
    # r = (steer_torque, pedaling_torque)

    p = constants

    permutation = [0, 1, 2, 3, 7, 4, 5, 6, 11, 12, 13, 8, 14, 9, 10, 15]

    # x = x[[0,1,2,3,7,4,5,6,11,12,13,8,14,9,10,15]]
    x_reordered = sm.Matrix([x.row(i) for i in permutation])
    x = x_reordered.as_immutable()

    q1, q2, q3, q4, q5, q6, q7, q8,  = x[:8]
    u1, u2, u3, u4, u5, u6, u7, u8 = x[8:]

    nh_cons = system.nonholonomic_constraints
    h_cons = system.holonomic_constraints
    kdes = system.kdes

    eoms = eoms.col_join(sm.Matrix(kdes)).col_join(sm.Matrix(h_cons)).col_join(sm.Matrix(nh_cons))

    # disturbance = sm.Matrix([disturbance])

    return t, x, r, eoms, p, bicycle


def export_constants(constants: dict[str, float]) -> None:
    """
    Export the constants to a pickle file for later use.

    Parameters
    ----------
    constants : Dictionary of {symbol: value} for constant parameters
    """

    if platform.system() == "Windows":
        full_file_name = f"model_files\constants_d.pkl"
    else:
        full_file_name = f"model_files/constants_d.pkl"

    with open(full_file_name, "wb") as f:
        pickle.dump(constants, f)


if __name__ == "__main__":

    pass

    config = {'roll_control' : True,
              'steer_torque' : True
        }

    model_name = 'model_0'
    t, x, r, eoms, p, bicycle = generate_model(model_name, config)

    # export_constants(constants)
