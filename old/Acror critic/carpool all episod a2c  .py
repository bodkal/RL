import os
os.environ['LANG']='en_US.UTF-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import numpy as np
import tensorflow as tf







class Network:
    def __init__(self, state_size, action_size, learning_rate,discount_factor):
        self.state_size=state_size
        self.action_size=action_size
        self.learning_rate=learning_rate
        self.create_actor()
        self.create_critic()
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.Huber()

    def create_actor(self):
        self.actor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(units=24, activation='elu', kernel_initializer=tf.keras.initializers.HeUniform()),
            tf.keras.layers.Dense(units=12, activation='elu', kernel_initializer=tf.keras.initializers.HeUniform()),

            tf.keras.layers.Dense(units=self.action_size, activation='softmax')
        ])

    def create_critic(self):
        self.critic = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(units=24, activation='elu', kernel_initializer=tf.keras.initializers.HeUniform()),
            tf.keras.layers.Dense(units=12, activation='elu', kernel_initializer=tf.keras.initializers.HeUniform()),

            tf.keras.layers.Dense(units=1, activation='linear')
        ])

class A2C(Network):
    def __init__(self,env, learning_rate,discount_factor):
        super().__init__(env.observation_space._shape[0],env.action_space.n, learning_rate,discount_factor)
        self.discount_factor =discount_factor  # Discount factor for past rewards
        self.max_steps_per_episode = 10000
        self.env=env



    def trine(self):
        action_probs_history = []
        critic_value_history = []
        rewards_history = [0]
        running_reward = 0
        episode_count = 0
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        while True:  # Run until solved
            state = env.reset()
            episode_reward = 0
            with tf.GradientTape(persistent=True) as tape:
                while True:
                    # env.render(); Adding this line would show the attempts
                    # of the agent in a pop up window.

                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action_probs = self.actor(state)
                    critic_value =  self.critic(state)
                    critic_value_history.append(critic_value[0, 0])

                    # Sample action from action probability distribution
                    action = np.random.choice(self.action_size, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))

                    # Apply the sampled action in our environment
                    state, reward, done, _ = env.step(action)
                    rewards_history.append(reward+self.discount_factor*rewards_history[-1])
                    episode_reward += reward

                    if done:
                        break
                    env.render()

                # Update running reward to check condition for solving
                #running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                running_reward+=episode_reward
                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic

                # Normalize
                returns = np.array(rewards_history[1:][::-1])
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                       self.loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

            self.update_net( tape, actor_losses, critic_losses)

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history=[0]

            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward/10, episode_count))
                running_reward=0

            if running_reward/10 > 475:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

    def update_net(self,tape,actor_losses,critic_losses):
        # Backpropagation
        actor_grads = tape.gradient(actor_losses, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_losses, self.critic.trainable_variables)

        self.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

if __name__=="__main__":
    # Configuration parameters for the whole setup

    # Create the environment
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 500
    # Set seed for experiment reproducibility
    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    state_size = 4
    action_size = env.action_space.n
    learning_rate=0.001
    discount_factor=0.99

    model=A2C(env,learning_rate,discount_factor)
    model.trine()