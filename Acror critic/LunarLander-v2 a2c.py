import os

os.environ['LANG'] = 'en_US.UTF-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

class MemoryHolder():
    def __init__(self):
        self.rewards=[]
        self.action_probs=[]
        self.critic_value=[]

    def save_semple(self,rewards,action_probs,critic_value):
        self.rewards.append(rewards)
        self.action_probs.append(action_probs)
        self.critic_value.append(critic_value)

    def get_sum_reword_by_step(self,discount_factor,eps):
        returns = []
        for r in self.rewards[::-1]:
            discounted_sum = r + discount_factor * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = th.tensor(returns, dtype=th.float32)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

class ActorCustomLoss(nn.Module):
    def __init__(self):
        super(ActorCustomLoss, self).__init__()

    def forward(self, log_prob, returns, value):
        diff = returns - value
        return -log_prob * diff


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size,hidden_layer, learning_rate):
        super(ActorNet, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.fc3 = nn.Linear(hidden_layer//2, self.action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate);
        self.loss_fn = ActorCustomLoss()
        self.leg_in_ground = 0

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = th.softmax(self.fc3(x), dim=-1)
        return x


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size,hidden_layer, learning_rate):
        super(CriticNet, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.fc3 = nn.Linear(hidden_layer//2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate);
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class A2C:
    def __init__(self, env_name,rander,max_episode_steps, learning_rate, discount_factor,hidden_layer, seed):

        if rander:
            self.env = gym.make(env_name, render_mode="human")
        else:
            self.env = gym.make(env_name)

        state_size = self.env.observation_space._shape[0]
        action_size = self.env.action_space.n

        self.actor = ActorNet(state_size, action_size,hidden_layer, learning_rate)
        self.critic = CriticNet(state_size, action_size,hidden_layer, learning_rate)

        self.discount_factor = discount_factor  # Discount factor for past rewards
        self.max_steps_per_episode = 10000
        self.rander=rander
        # env = gym.make("LunarLander-v2", render_mode="human")


        self.env._max_episode_steps = max_episode_steps

        self.seed = seed

    def train(self):
        action_probs_history = []
        critic_value_history = []
        rewards_history = [0]
        running_reward = 0
        episode_count = 0
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        #while True:  # Run until solved
        for i in range(1000):
            state = self.env.reset(seed=self.seed)[0]
            episode_reward = 0
            while True:

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # env.render() #; Adding this line would show the attempts
                # of the agent in a pop up window.

                state = th.tensor(state, dtype=th.float32)
                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs = self.actor.forward(state)
                critic_value = self.critic.forward(state).squeeze()

                ##critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())

                # action_probs_history.append(th.math.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                state, reward, done, _, _ = self.env.step(action)

                rewards_history.append(reward)
                action_probs_history.append(th.log(action_probs[action]))
                critic_value_history.append(critic_value)

                episode_reward += reward

                if self.is_done(state, done):
                    break

                if self.rander:
                    self.env.render()

            # Update running reward to check condition for solving
            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward += episode_reward
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + self.discount_factor * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = th.tensor(returns, dtype=th.float32)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            self.update_net(zip(action_probs_history, critic_value_history, returns))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history = [0]

            # Log details
            episode_count += 1
            if episode_count % 25 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward / 10, episode_count))
                running_reward = 0

            if running_reward / 25 > 475:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

    def get_loss(self, history):
        # Calculating loss values to update our network

        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.

            actor_losses.append(self.actor.loss_fn(log_prob, ret, value))  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(self.critic.loss_fn(value, ret))

        return actor_losses, critic_losses

    def update_net(self, history):
        actor_losses, critic_losses = self.get_loss(history)

        # Zero the gradients for actor optimizer
        self.actor.optimizer.zero_grad()
        # Sum up and perform the backward pass for actor losses
        actor_loss = th.stack(actor_losses).sum()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Zero the gradients for critic optimizer
        self.critic.optimizer.zero_grad()
        # Sum up and perform the backward pass for critic losses
        critic_loss = th.stack(critic_losses).sum()
        critic_loss.backward()
        self.critic.optimizer.step()

    def is_done(self, state, done):

        if state[-1] > 0 and state[-2] > 0:
            self.leg_in_ground += 1;
        else:
            self.leg_in_ground = 0;

        # print(state[-1], state[-2], self.leg_in_ground)

        if done:
            return True

        if self.leg_in_ground >= 5:
            return True

        return False


if __name__ == "__main__":
    # Configuration parameters for the whole setup

    # Create the environment
    env_name = "LunarLander-v2"
    rander = False
    max_episode_steps=500
    # Set seed for experiment reproducibility
    seed_ = 42
    # th.random.seed(seed_)
    np.random.seed(seed_)

    learning_rate = 0.001
    discount_factor = 0.99

    for hidden_layer in [82,120,160,250]:
        for i in range(3):
            print(f"i: {i} hidden_layer: {hidden_layer}")
            model = A2C(env_name,rander,max_episode_steps, learning_rate, discount_factor, hidden_layer,seed_)
            model.train()
