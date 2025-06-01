import math

import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state]) #按照当前R值为均值根据高斯密度函数得到随机奖励
        cumProb = np.cumsum(self.mdp.T[action,state,:])#把
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''
        qLearning算法，需要将Epsilon exploration和 Boltzmann exploration 相结合。
        以epsilon的概率随机取一个动作，否则采用 Boltzmann exploration取动作。
        当epsilon和temperature都为0时，将不进行探索。

        Inputs:
        s0 -- 初始状态
        initialQ -- 初始化Q函数 (|A|x|S| array)
        nEpisodes -- 回合（episodes）的数量 (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- 每个回合的步数(steps)
        epsilon -- 随机选取一个动作的概率
        temperature -- 调节 Boltzmann exploration 的参数

        Outputs: 
        Q -- 最终的 Q函数 (|A|x|S| array)
        policy -- 最终的策略
        rewardList -- 每个episode的累计奖励（|nEpisodes| array）
        '''
        Q = initialQ.copy()  # 初始化Q函数
        rewardList = []  # 初始化累计奖励列表

        for episode in range(nEpisodes):
            s = s0  # 将当前状态设置为初始状态
            episodeReward = 0  # 初始化该回合的累计奖励

            for step in range(nSteps):
                # 选择动作
                if np.random.rand() < epsilon:
                    a = np.random.randint(self.mdp.nActions)  # 随机选择动作
                else:
                    if temperature > 0:
                        prob = np.exp(Q[:, s] / temperature)  # 使用Boltzmann探索计算动作概率
                        prob /= np.sum(prob)
                        a = np.random.choice(range(self.mdp.nActions), p=prob)  # 根据概率选择动作
                    else:
                        a = np.argmax(Q[:, s])  # 贪婪选择动作

                # 采样奖励和下一个状态
                reward, nextState = self.sampleRewardAndNextState(s, a)

                # 更新Q函数
                learningRate = 1.0 / (1.0 + step)  # 调整学习率
                Q[a, s] = Q[a, s] + learningRate * (reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[a, s])

                # 更新累计奖励
                episodeReward += reward

                # 更新当前状态
                s = nextState

            rewardList.append(episodeReward)

        # 计算策略
        policy = np.argmax(Q, axis=0)

        return [Q,policy,rewardList]