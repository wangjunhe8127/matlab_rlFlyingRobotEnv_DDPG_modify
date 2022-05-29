mdl = 'rlFlyingRobotEnv';
integratedMdl = 'IntegratedFlyingRobot'; 
[~,agentBlk,observationInfo,actionInfo] = createIntegratedEnv(mdl,integratedMdl);
load('agent.mat')
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
experience = sim(env,agent,simOptions);