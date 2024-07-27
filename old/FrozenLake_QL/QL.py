# frozen-lake-ex1.py
import gym
import numpy as np

class QL:
    def __init__(self,env):
        # number of state
        self.n_observations = env.observation_space.n

        # number of action
        self.n_actions = env.action_space.n

        #Q for stocastic invrment
        self.Q_table = np.zeros((self.n_observations, self.n_actions))

        self.n_episodes=10000
        self.max_iter_episode=100

        # discounted factor
        self.gamma = 0.99

        self.rewards_per_episode = []
        self.e_greedy=0.1
        self.larning_rate=0.1
        # minimum of exploration proba
        self.min_exploration_proba = 0.01
        self.exploration_decreasing_decay=0.001

    def run_over_episod(self):
        for episod in range(self.n_episodes):
            print(episod)
            # we initialize the first state of the episode
            current_state = env.reset()
            done = False

            # sum the rewards that the agent gets from the environment
            total_episode_reward = 0
            for i in range(self.max_iter_episode):
                if np.random.uniform(0,1)<self.e_greedy:
                    action =env.action_space.sample()
                else:
                    action = np.argmax(self.Q_table[current_state,:])
                next_state, reward, _ , _ = env.step(action)

                self.Q_table[current_state,action]=(1-self.larning_rate)*self.Q_table[current_state,action]+\
                                                   self.larning_rate*(reward+self.gamma*max(self.Q_table[next_state]))
                total_episode_reward = total_episode_reward + reward
                current_state=next_state
            self.rewards_per_episode.append(total_episode_reward)
            self.e_greedy=max(self.min_exploration_proba,np.exp(-self.exploration_decreasing_decay*episod))

    def evluate(self):
        print("Mean reward per thousand episodes")
        index= int(self.n_episodes/10)
        for i in range(10):
            print(f"{index*(i+1)} : meanespiode reward: {np.mean(self.rewards_per_episode[index*i: index*(i+1)])}")


env = gym.make("FrozenLake-v1")
env.reset()
env.render()
model=QL(env)
model.run_over_episod()
model.evluate()