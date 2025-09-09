# OpenLoopBalanceControl
Code and Data to run open loop balance control identification using direct collocation method
## Context and Objectives
- The torque based cyclist balance strategy is poorly describe in the litrature, need for observation, quantification and description
- The link between cyclist control actions and workload is still unknown, gaze based workload metrics and balance performance metrics can be used as workload proxy
- This study aims to estimates cyclists control actions on constant heading task under two condition: hand-on and hand-off
- Then quantified, the amount of control actions will tested as correlated with workload proxy

Question: what characteristics of control actions is related to workload and least performance?
- Energy, power, work, variability, power spectrum....

## TO DO
- Include head accelerations to the model
- Add some bounds (currently have none)?
- Scale models to participants
- Run optim, fix millions of bugs?
- Post processing the control actions (deciding how to describe control actions)
- Run the stats
