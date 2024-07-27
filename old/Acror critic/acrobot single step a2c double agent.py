import os
# os.environ['LANG']='en_US.UTF-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing

class Network:
    def __init__(self, state_size, action_size, learning_rate,discount_factor):
        self.state_size=state_size
        self.action_size=action_size
        self.learning_rate=learning_rate
        self.create_actor()
        self.create_critic()
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_loss = tf.keras.losses.MeanSquaredError()
        self.init_actor_weights = self.actor.get_weights()
        self.init_critic_weights = self.critic.get_weights()


    def create_actor(self):
        self.actor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(units=12, activation='elu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
            #tf.keras.layers.Dense(units=16, activation='elu', kernel_initializer=tf.keras.initializers.GlorotUniform()),

            tf.keras.layers.Dense(units=self.action_size, activation='softmax')
        ])

    def create_critic(self):
        self.critic = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(units=64, activation='elu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
           # tf.keras.layers.Dense(units=16, activation='elu', kernel_initializer=tf.keras.initializers.GlorotUniform()),

            tf.keras.layers.Dense(units=1, activation='linear')
        ])

    def critic_loss(self, vlaue,reword):
        loss=tf.keras.metrics.mean_squared_error(reword,vlaue)
        return loss

    def actor_loss(self, prob, action, td_error):
       # loss=-tf.math.log(prob[0, action]) * td_error
        all_action=[0]*self.action_size
        all_action[action]=1
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=all_action,logits=prob)*td_error)
        return loss

    def update_net(self, a_tape, c_tape, old_value, td_target, prob, action, td_error):
        # Backpropagation
        actor_grads = a_tape.gradient(self.actor_loss(prob, action, td_error), self.actor.trainable_variables)
        critic_grads = c_tape.gradient(self.critic_loss(td_target,old_value), self.critic.trainable_variables)

        self.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def restrat_net(self):
        self.actor.set_weights(self.init_actor_weights)
        self.critic.set_weights(self.init_critic_weights)

class A3C(Network):
    def __init__(self,env,env1,acshion_size,state_size, learning_rate,discount_factor):
        super().__init__(state_size,acshion_size, learning_rate,discount_factor)
        self.discount_factor =discount_factor  # Discount factor for past rewards
        self.max_steps_per_episode = env._max_episode_steps+1
        self.env1=env
        self.env2=env1


        self.running_reward1 = []
        self.running_reward2 = []

    def print_reword(self):
        plt.plot(self.running_reward1)
        plt.plot(self.running_reward2)

        plt.show()

    def make_step(self,action,env):
        state, reward, done, _ =env.step(action)
        state = tf.expand_dims(state, 0)

        return state, reward, done


    def trine(self):
        episode_count = 0
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        while True:  # Run until solved
            state1 = self.env1.reset()
            state1 = tf.expand_dims(state1, 0)

            state2 = self.env2.reset()
            state2 = tf.expand_dims(state2, 0)

            episode_reward1 = 0
            episode_reward2 = 0

            I = 1
            for i in range(self.max_steps_per_episode):
                with tf.GradientTape() as tape_actor1, tf.GradientTape() as tape_critic1, tf.GradientTape() as tape_actor2, tf.GradientTape() as tape_critic2:

                    self.env1.render()
                    self.env2.render()

                    action_probs1 = self.actor(state2)
                    old_value1 =  self.critic(state2)

                    action_probs2 = self.actor(state2)
                    old_value2 = self.critic(state2)

                    # Sample action from action probability distribution
                    action1 = np.random.choice(self.action_size, p=np.squeeze(action_probs1))
                    action2 = np.random.choice(self.action_size, p=np.squeeze(action_probs2))

                    # Apply the sampled action in our environment
                    next_state, reward1, done1 = self.make_step(action1,self.env1)
                    new_value1 =  self.critic(next_state)

                    next_state, reward2, done2 = self.make_step(action2,self.env2)
                    new_value2 =  self.critic(next_state)

                    td_target1=reward1+self.discount_factor*new_value1
                    td_error1 = td_target1-old_value1

                    td_target2=reward2+self.discount_factor*new_value2
                    td_error2 = td_target2-old_value2

                    self.update_net(tape_actor1,tape_critic1,old_value1,td_target1,action_probs1,action1,td_error1)
                    self.update_net(tape_actor2,tape_critic2,old_value2,td_target2,action_probs2,action2,td_error2)

                    episode_reward1 += reward1
                    episode_reward2 += reward2

                    if done1 or done2:
                        break
                    state = next_state
            self.running_reward1.append(episode_reward1)
            self.running_reward2.append(episode_reward2)

           # self.restrat_net()

            episode_count += 1
            if episode_count % 1 == 0:
                template = "mean reward: {:.2f} at episode {}"
                print(template.format(np.mean(self.running_reward1[-100:]+self.running_reward2[-100:]), episode_count))
            #tmp_all_reword=running_reward[:-100]

            if episode_count>100 and np.mean(self.running_reward1[-100:]+self.running_reward2[-100:]) > -85:
                print("Solved at episode : {}, at reword of {}".format(episode_count,np.mean(self.running_reward1[-100:]+self.running_reward2[-100:])))
                self.print_reword()
                break





if __name__=="__main__":
    # Configuration parameters for the whole setup

    # Create the environment
    env1 = gym.make("Acrobot-v1")
    env2 = gym.make("Acrobot-v1")

    #env._max_episode_steps = 100
    # Set seed for experiment reproducibility
    # seed = 1
    # env1.seed(seed)
    # tf.random.set_seed(seed)
    # np.random.seed(seed)

    action_size = env1.action_space.n
    learning_rate=0.001
    discount_factor=0.99
    acshion_size=3
    state_size=6
    model=A3C(env1,env2,acshion_size,state_size,learning_rate,discount_factor)
    model.trine()