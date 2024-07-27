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
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=all_action,logits=prob)* td_error)
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

class A2C(Network):
    def __init__(self,env,acshion_size,state_size, learning_rate,discount_factor):
        super().__init__(state_size,acshion_size, learning_rate,discount_factor)
        self.discount_factor =discount_factor  # Discount factor for past rewards
        self.max_steps_per_episode = env._max_episode_steps+1
        self.env=env
        self.scaler=self.normlize()
        self.running_reward = []

    def print_reword(self):
        plt.plot(self.running_reward)
        plt.show()

    def make_step(self,action):
        state, reward, done, _ = self.env.step(action)
        #state = tf.expand_dims(state, 0)
        state= tf.expand_dims(np.squeeze(self.scaler.transform([state])),0)

        return state, reward, done

    def normlize(self):
        state_space_samples = np.array([self.env.observation_space.sample() for x in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(state_space_samples)
        return scaler

    def trine(self):
        episode_count = 0
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        while True:  # Run until solved
            state = self.env.reset()
            state = tf.expand_dims(state, 0)

            episode_reward = 0
            I = 1
            for i in range(self.max_steps_per_episode):
                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:

                    self.env.render()

                    action_probs = self.actor(state)
                    old_value =  self.critic(state)

                    # Sample action from action probability distribution
                    action = np.random.choice(self.action_size, p=np.squeeze(action_probs))

                    action_value = 1 if action==1 else -1

                    # Apply the sampled action in our environment
                    next_state, reward, done = self.make_step([action_value])
                    new_value =  self.critic(next_state)

                    td_target=reward+self.discount_factor*new_value
                    td_error = td_target-old_value

                    self.update_net(tape_actor,tape_critic,old_value,td_target,action_probs,action,td_error)

                    episode_reward += reward

                    if done:

                        break
                    state = next_state
            self.running_reward.append(episode_reward)

           # self.restrat_net()

            episode_count += 1
            if episode_count % 1 == 0:
                template = "mean reward: {:.2f} at episode {}"
                print(template.format(np.mean(self.running_reward[-100:]), episode_count))
            #tmp_all_reword=running_reward[:-100]

            if episode_count>100 and np.mean(self.running_reward[-100:]) > 90:
                print("Solved at episode : {}, at reword of {}".format(episode_count, np.mean(self.running_reward[-100:])))
                self.print_reword()
                break





if __name__=="__main__":
    # Configuration parameters for the whole setup

    # Create the environment
    env = gym.make("MountainCarContinuous-v0")
    #env._max_episode_steps = 100
    # Set seed for experiment reproducibility
    seed = 1
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    learning_rate=0.0001
    discount_factor=0.99
    acthion_size=2
    state_size=2

    model=A2C(env,state_size,acthion_size,learning_rate,discount_factor)
    model.trine()