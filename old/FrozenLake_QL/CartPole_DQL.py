# import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow as tf
# frozen-lake-ex1.py
import gym
import numpy as np
import random

class QL:
    def __init__(self,env):
        self.env=env
        self.n_episodes=10000
        self.max_iter_episode=501

        # discounted factor
        self.discounted = 0.95

        self.rewards_per_episode = []
        self.e_greedy=1
        self.larning_rate=0.002
        self.cainge_net=99
        self.memory_size=10000
        self.sempel_size=150
        self.exploration_decreasing_decay=0.9995

        self.old_state = np.array([env.reset()])
        self.old_next_state = np.copy(self.old_state)
        self.old_action = np.zeros(1)
        self.old_reward = np.zeros(1)
        self.old_done =  np.zeros(1)

        self.net_online=self.get_net()
        self.net_offline=keras.models.clone_model(self.net_online)

        self.rewards_per_episode=[]

    def get_net(self):
        net = keras.Sequential([keras.layers.InputLayer(input_shape=(4,)),
                                keras.layers.Dense(units=16,activation='relu',kernel_initializer=keras.initializers.HeUniform()),
                                keras.layers.Dense(units=64, activation='relu',kernel_initializer=keras.initializers.HeUniform()),
                                keras.layers.Dense(units=16,activation='relu',kernel_initializer=keras.initializers.HeUniform()),
                                keras.layers.Dense(units=2,activation='linear')])
        net.compile(optimizer=keras.optimizers.SGD(learning_rate=self.larning_rate),loss='mse')
        return net

    def get_action(self,state):
        if np.random.uniform(0, 1) < self.e_greedy:
            return self.env.action_space.sample()
        return self.net_online.predict(np.atleast_2d(state)).argmax()

    def remember(self, state, action, reward, next_state, done):
        self.old_state=np.vstack((self.old_state,[state]))[-self.memory_size:]
        self.old_next_state=np.vstack((self.old_next_state,[next_state]))[-self.memory_size:]
        self.old_action=np.hstack((self.old_action,action))[-self.memory_size:].astype(int)
        self.old_reward=np.hstack((self.old_reward,reward))[-self.memory_size:]
        self.old_done=np.hstack((self.old_done,done))[-self.memory_size:]


    def get_random_sempal_from_past(self):
        if self.old_state.__len__()>self.sempel_size:
            sampel=np.random.randint(0,self.old_state.__len__(),self.sempel_size)
        else:
            sampel=range(self.old_state.__len__())
        return self.old_state[sampel],self.old_next_state[sampel],self.old_action[sampel],self.old_reward[sampel] ,self.old_done[sampel]

    def get_batch_from_memory(self):
        return random.sample(self.memory, self.batch_size)

    def get_update_q_val(self):
        sampel_state, sampel_next_state, sampel_action, sampel_reward, sampel_done = self.get_random_sempal_from_past()

        predict_state = np.copy(self.net_online.predict(sampel_state))
        predict_state_next = self.net_offline.predict(sampel_next_state)

        predict_state[range(sampel_action.__len__()),sampel_action] = sampel_reward \
                                        + (sampel_done == False) * self.discounted *np.amax(predict_state_next,axis=1)


        return sampel_state,predict_state

    def run_over_episod(self):
        env.reset()
        env.render()
        current_state = self.old_state[-1]
        for episod in range(self.n_episodes):
            # we initialize the first state of the episode

            # sum the rewards that the agent gets from the environment
            total_episode_reward = 0
            for i in range(self.max_iter_episode):
                action=self.get_action(current_state)

                next_state, reward, done , _ = self.env.step(action)
                total_episode_reward+=reward
                self.env.render()

                self.remember(current_state,action,reward,next_state,done)

                sampel_old_state, update_q_val=self.get_update_q_val()

                self.net_online.train_on_batch(sampel_old_state,update_q_val)

                if not i%self.cainge_net:
                    self.net_offline.set_weights(self.net_online.get_weights())

                self.e_greedy=max(self.e_greedy*self.exploration_decreasing_decay,0.01)
                if done:
                    break
                current_state=next_state

            print(f"episod - {episod},\treward - {total_episode_reward}\te_greedy - {self.e_greedy}")
            self.env.reset()
            self.rewards_per_episode.append(total_episode_reward)


    def evluate(self):
        print("Mean reward per thousand episodes")
        index= int(self.n_episodes/10)
        for i in range(10):
            print(f"{index*(i+1)} : meanespiode reward: {np.mean(self.rewards_per_episode[index*i: index*(i+1)])}")


if __name__=='__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.envs.make("CartPole-v1")
    model=QL(env)
    net=model.get_net()
    model.run_over_episod()
    model.evluate()
    model.get_path()