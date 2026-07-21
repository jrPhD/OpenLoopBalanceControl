# OpenLoopBalanceControl
Code and Data to perform trajectories optimization from bicycle (+rider) motion and get the control troques of the rider.

## Context and Objectives
- The torque based cyclist balance strategy is poorly describe in the litrature, need for observation, quantification and description
- The link between cyclist control actions and workload is still unknown, gaze based workload metrics and balance performance metrics can be used as workload proxy

## Aim of this code
This code is meant to be a open tool to estimate cyclist's control torques from bicycle motion.
Sources of motion can be diverse (mocap, IMU, GPS etc.).
Complexity of the model can be adjusted : rigid rider, leaning rider, more advanced...

## Analysis to do
- impact of model complexity on tracking performance and accuracy (real data + synthetic data)
- What is a/the good/best performance metric for this method?
- sensitivity analysis to bicycle and rider parameters
- IMU to torques ? What's the best setting ? 
- compare torque control metrics to gaze metrics

## TO DO
- make a rigid rider model -> Done
- generate synthetic data with a leaning rider, ie: we know the torques and the motion and we test our method
- run the method on real data :
- hands-off cycling, do we get a very low steer torque? -> We get full steer torque, most controllabe input
- data with steer torque sensor, compute accuracy
- data with chest cast, do we get a very low roll torque?
- find correlations between accuracy and tracking performance
- run Morris analysis with bicycle parameters

