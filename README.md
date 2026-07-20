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
- Include head accelerations to the model
- Pin or Spherical joint for the head? 
- Add some bounds (currently have none)?
- Maybe improve the initial guess, have you ever thougth of doing some basic regressions from previously solved similar problems?
- Scale models to participants
- Run optim, fix millions of bugs?
- Post processing the control actions (deciding how to describe control actions)
- Run the stats

