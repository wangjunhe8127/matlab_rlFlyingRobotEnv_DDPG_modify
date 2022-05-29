mdl = 'rlFlyingRobotEnv';
open_system(mdl); %打开环境env
theta0 = 0; %初始角度=0
x0 = -15; %初始x
y0 = 0; %初始y
Ts = 0.4; %采样时间0.4s
Tf = 30; %仿真总时间
%飞行机器人模型的 Simulink 模型IntegratedEnv，该模型以闭环方式连接到代理块
integratedMdl = 'IntegratedFlyingRobot'; 
% %路径返回到新集成模型中的 RL Agent 块，单个连续观察规范的对象，动作规范的对象
[~,agentBlk,observationInfo,actionInfo] = createIntegratedEnv(mdl,integratedMdl);
%7个状态数据 总共是1*7=7个
numObs = prod(observationInfo.Dimension);
%状态变量名
observationInfo.Name = 'observations';
%两个动作数据 总共1*2=2个
numAct = prod(actionInfo.Dimension);
%动作范围
actionInfo.LowerLimit = -ones(numAct,1);
actionInfo.UpperLimit =  ones(numAct,1);
%动作变量名
actionInfo.Name = 'thrusts';
%将动作，变量以及生成env变量
env = rlSimulinkEnv(integratedMdl,agentBlk,observationInfo,actionInfo);
%将simulink中输入量reset
env.ResetFcn = @(in)flyingRobotResetFcn(in);
%提供随机数种子
rng(0) 
% Specify the number of outputs for the hidden layers.
hiddenLayerSize = 100; %隐含层数量
%
observationPath = [
    %观测数据维度，标准化，命名为observation层
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    %隐层，命名为fc1
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    %relu层，命名为relu1
    reluLayer('Name','relu1')
    %再次隐含层，命名为fc2
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')
    %将多层相加
    additionLayer(2,'Name','add')
    %relu层，命名为relu2
    reluLayer('Name','relu2')
    %再次隐含层，命名为fc3
    fullyConnectedLayer(hiddenLayerSize,'Name','fc3')
    %relu层，命名为relu3
    reluLayer('Name','relu3')
    %最后一层，命名为fc4
    fullyConnectedLayer(1,'Name','fc4')];

actionPath = [
    %动作数据维度，标准化，命名为action层
    featureInputLayer(numAct,'Normalization','none','Name','action')
    %隐层，命名为fc5
    fullyConnectedLayer(hiddenLayerSize,'Name','fc5')];

% 构建图
criticNetwork = layerGraph(observationPath);
% 增加一个分支
criticNetwork = addLayers(criticNetwork,actionPath);

% 构建critic网络，由于前面critic中有个additionalLayer，其第一个输入来自上一层，第二个输入来自fc5，也就是动作
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');
% 构建critic参数
criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
%构建critic Observation表示状态输入层，Action表示网络动作输出
critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
%构建actor网络
actorNetwork = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(numAct,'Name','fc4')
    tanhLayer('Name','tanh1')];
%构建actor参数
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
%构建actor
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'tanh1'},actorOptions);
%构建智能体参数
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6 ,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',256);
agentOptions.NoiseOptions.Variance = 1e-1;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-6; 
%构建智能体
agent = rlDDPGAgent(actor,critic,agentOptions);% 智能体网络
maxepisodes = 1000;
maxsteps = ceil(Tf/Ts);
%构建训练参数
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'StopOnError',"on",...
    'Verbose',false,...
    'Plots',"training-progress",...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',415,...
    'ScoreAveragingWindowLength',10,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',415); 
doTraining = true; 
if doTraining       
    % 开始训练
    trainingStats = train(agent,env,trainingOptions);
else
    % Load the pretrained agent for the example.
    load('agent')       
end
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
save agent
experience = sim(env,agent,simOptions);