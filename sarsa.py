import numpy as np
import argparse
import gym
import time

np.random.seed(2)

class SARSA:

    def __init__(self, n_states, n_actions, lr=0.01, epsilon=0.2, gamma=0.9):
        self._LEARNING_RATE = lr
        self._GAMMA = gamma
        self._EPSILON = epsilon
        self._ACTION = n_actions
        self._STATES = n_states
        self._q_table = np.random.uniform(high=0.5, size=(self._STATES, self._ACTION))

    @property
    def q_table(self):
        return self._q_table

    @property
    def LEARNING_RATE(self):
        return self._LEARNING_RATE

    @LEARNING_RATE.setter
    def LEARNING_RATE(self, value):
        assert isinstance(value, int) or isinstance(value, float)
        self._LEARNING_RATE = value

    @property
    def GAMMA(self):
        return self._GAMMA

    @GAMMA.setter
    def GAMMA(self, value):
        assert value < 1 and value > 0
        self._GAMMA = value

    @property
    def EPSILON(self):
        return self._EPSILON

    @EPSILON.setter
    def EPSILON(self, value):
        assert value < 1 and value > 0
        self._EPSILON = value

    @property
    def STATES(self):
        return self._STATES

    @property
    def ACTIONS(self):
        return self._ACTION

    '''
    Take Action:
        select action with epsilon greedy
    param:
        state: current state
    return:
        action number
    '''
    def takeAction(self, state: int) -> int:
        action = None
        if np.random.uniform() < self._EPSILON:
            action = np.random.choice(self._ACTION)
        else:
            action = np.argmax(self._q_table[state])
        return action

    '''
    learning:
        use bellman equation and TD to update q table in every episode
    sarsa is on-policy since we update our q tablb with TD target using 
    next action chosen based on behavior policy
    '''
    def learning(self, state, next_state, action, reward, done):
        TD_target = reward
        if not done:
            next_action = self.takeAction(next_state)
            TD_target += self._GAMMA * self._q_table[next_state][next_action]
        TD_error = TD_target - self._q_table[state][action]
        self._q_table[state][action] += self._LEARNING_RATE * TD_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input parameters: LEARNING_RATE, GAMMA, EPSILON, STATES, MAX_STEP, EPISODE")
    parser.add_argument('-g', "--gamma", type=float, default=0.9, help="discount factor, float type, default 0.9")
    parser.add_argument('-lr', "--learningrate", type=float, default=0.01, help="learning rate, float type, default 0.01")
    parser.add_argument('-eg', "--epsilon", type=float, default=0.1, help="epsilon greedy factor, float type, default 0.1")
    parser.add_argument('-ep', "--episode", type=int, default=5000, help="maximum episode, int type, default 1000")
    parser.add_argument('-gm', "--game", type=str, default='Taxi-v3', help="choose game, string type, default cartpole")
    args = parser.parse_args()

    env = gym.make(args.game)
    agent = SARSA(env.observation_space.n, env.action_space.n, args.learningrate, args.epsilon, args.gamma)

    for eps in range(args.episode):
        state = env.reset()
        cnt = 0
        while True:
            cnt += 1
            env.render()
            action = agent.takeAction(state)
            next_state, reward, done, info = env.step(action)
            agent.learning(state, next_state, action, reward, done)

            state = next_state
            #time.sleep(1)

            if done:
                print("episode {} finished in {} step(s)".format(eps, cnt))
                break

    cnt = 0
    state = env.reset()
    while True:
        cnt += 1
        env.render()
        action = agent.takeAction(state)
        next_state, reward, done, info = env.step(action)
        agent.learning(state, next_state, action, reward, done)

        state = next_state

        time.sleep(1)

        if done:
            print("playing finished in {} step(s)".format(cnt))
            break

    print(agent.q_table)
    env.close()
