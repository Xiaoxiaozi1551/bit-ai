import numpy as np
import MDP
from sympy import *

class RL2:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
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

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def sampleSoftmaxPolicy(self, policyParams, state):
        '''从随机策略中采样单个动作的程序，通过以下概率公式采样
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        本函数将被reinforce()调用来选取动作

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action

        提示：计算出概率后，可以用np.random.choice()，来进行采样
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        action = 0
        probabilities = np.exp(policyParams[:, state]) / np.sum(np.exp(policyParams[:, state]))
        action = np.random.choice(self.mdp.nActions, p=probabilities)

        return action



    def epsilonGreedyBandit(self, nIterations):
        '''Epsilon greedy 算法 for bandits (假设没有折扣因子).
        Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []
        s = 0

        epsilon = 1 / nIterations
        nVisits = np.zeros(self.mdp.nActions)
        for i in range(nIterations):
            if np.random.rand() < epsilon:
                # Explore: choose a random action
                action = np.random.choice(self.mdp.nActions)
            else:
                # Exploit: choose the action with the highest empirical mean
                action = np.argmax(empiricalMeans)

            reward, _ = self.sampleRewardAndNextState(0, action)
            reward_list.append(reward)

            # Update the empirical mean and visit count for the chosen action
            nVisits[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / nVisits[action]

        return empiricalMeans,reward_list

    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling 算法 for Bernoulli bandits (假设没有折扣因子)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards


        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)

        提示：根据beta分布的参数，可以采用np.random.beta()进行采样
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []
        s = 0

        for i in range(nIterations):
            sampled_means = np.random.beta(prior[:, 0], prior[:, 1])
            action = np.argmax(sampled_means)

            reward, _ = self.sampleRewardAndNextState(0, action)
            reward_list.append(reward)
            empiricalMeans[action] = (empiricalMeans[action] * i + reward) / (i + 1)

            prior[action, 0] += reward
            prior[action, 1] += 1 - reward

        return empiricalMeans,reward_list

    def UCBbandit(self, nIterations):
        '''Upper confidence bound 算法 for bandits (假设没有折扣因子)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []
        s = 0
        n_visits = np.zeros(self.mdp.nActions)

        for i in range(nIterations):
            if 0 in n_visits:
                action = np.where(n_visits == 0)[0][0]
            else:
                ucb_values = empiricalMeans + np.sqrt(2 * np.log(i + 1) / n_visits)
                action = np.argmax(ucb_values)

            reward, _ = self.sampleRewardAndNextState(0, action)
            reward_list.append(reward)
            n_visits[action] += 1
            empiricalMeans[action] = (empiricalMeans[action] * (n_visits[action] - 1) + reward) / n_visits[action]

        return empiricalMeans, reward_list

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
        rewardList = []
        policyParams = initialPolicyParams.copy()
        # 定义超参数
        gamma = 0.95
        alpha = 0.01

        for episode in range(nEpisodes):
            rewards = []
            states = []
            actions = []

            state = s0

            for step in range(nSteps):
                # 从策略中采样动作
                action = self.sampleSoftmaxPolicy(policyParams, state)

                # 采样奖励和下一个状态
                reward, nextState = self.sampleRewardAndNextState(state, action)

                # 保存当前状态、动作和奖励
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = nextState

            # 计算折扣奖励
            discountedRewards = np.zeros(nSteps)
            cumulativeReward = 0

            for t in reversed(range(nSteps)):
                cumulativeReward = rewards[t] + gamma * cumulativeReward
                discountedRewards[t] = cumulativeReward

            # 更新策略参数
            for t in range(nSteps):
                action = actions[t]
                state = states[t]

                # 计算策略梯度
                probabilities = np.exp(policyParams[:, state]) / np.sum(np.exp(policyParams[:, state]))
                gradient = np.zeros_like(policyParams)

                for a in range(self.mdp.nActions):
                    if a == action:
                        gradient[a, state] = 1 - probabilities[action]
                    else:
                        gradient[a, state] = -probabilities[a]

                # 更新策略参数
                policyParams += alpha * discountedRewards[t] * gradient

            # 计算累计折扣奖励并保存到rewardList中
            cumulativeReward = np.sum(rewards)
            rewardList.append(cumulativeReward)

        return [policyParams,rewardList]