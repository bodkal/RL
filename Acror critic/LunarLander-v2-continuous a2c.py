import os

os.environ['LANG'] = 'en_US.UTF-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ActorCustomLoss(nn.Module):
    def __init__(self):
        super(ActorCustomLoss, self).__init__()

    def forward(self, log_prob, returns, value):
        diff = returns - value
        return -log_prob * diff


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size,hidden_layer, learning_rate):
        super(ActorNet, self).__init__()
        logstds_param = nn.Parameter(th.full((action_size,), 0.1))
        self.register_parameter("logstds", logstds_param)

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
       # x = th.softmax(self.fc3(x), dim=-1)
        x = self.fc3(x)
        means = x
        stds = th.clamp(self.logstds.exp(), -10, 10)
        return th.distributions.Normal(means, stds)


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
    def __init__(self, env_name,rander,max_episode_steps,number_of_episode, learning_rate, discount_factor,hidden_layer, seed):

        if rander:
            self.env = gym.make("LunarLander-v2",
                                render_mode = "human",
                                continuous=True,
                                gravity=-10.0,
                                enable_wind=False,
                                wind_power=15.0,
                                turbulence_power=1.5,
                                max_episode_steps = max_episode_steps)
        else:
            self.env = gym.make("LunarLander-v2",
                                continuous=True,
                                gravity=-10.0,
                                enable_wind=False,
                                wind_power=15.0,
                                turbulence_power=1.5,
                                max_episode_steps = max_episode_steps)

        state_size = self.env.observation_space._shape[0]
        action_size = self.env.action_space.shape[0]

        self.actor = ActorNet(state_size, action_size,hidden_layer, learning_rate)
        self.critic = CriticNet(state_size, action_size,hidden_layer, learning_rate)

        self.discount_factor = discount_factor  # Discount factor for past rewards
        self.max_steps_per_episode = 10000

        self.number_of_episode = number_of_episode
        # env = gym.make("LunarLander-v2", render_mode="human")

        self.seed = seed

    def train(self):
        action_probs_history = []
        critic_value_history = []
        rewards_history = [0]
        running_reward = []
        #while True:  # Run until solved
        for episode_count in range( self.number_of_episode):

            if episode_count == 1000:
                self.env = gym.make("LunarLander-v2",
                                    render_mode="human",
                                    continuous=True,
                                    gravity=-10.0,
                                    enable_wind=False,
                                    wind_power=15.0,
                                    turbulence_power=1.5,
                                    max_episode_steps=max_episode_steps)
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
                action_norm_dists = self.actor.forward(state)
                actions = action_norm_dists.sample()
                action_probs_history.append(action_norm_dists.log_prob(actions))

                critic_value = self.critic.forward(state).squeeze()
                critic_value_history.append(critic_value)

                clip_actions = np.clip(actions, -1,1).detach().data.numpy()
                # Apply the sampled action in our environment
                state, reward, done, _, _ = self.env.step(clip_actions)

                rewards_history.append(reward)
                episode_reward += reward

                if self.is_done(state, done):
                    break

            # Update running reward to check condition for solving
            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward.append(episode_reward)
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic

            returns=self.calc_reword_for_episode(rewards_history)
            self.update_net(zip(action_probs_history, critic_value_history, returns))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history = [0]

            # Log details
            if ((episode_count+1) % 25) == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(np.mean(running_reward[:-10]), episode_count+1))

        return running_reward

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

    def calc_reword_for_episode(self,rewards_history):
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.discount_factor * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize
        returns = th.tensor(returns, dtype=th.float32)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

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


    def plot_resolt(self,rewards_lists,labels):
        # Loop through each rewards list and plot them
        for idx, rewards in enumerate(rewards_lists):
            plt.plot(rewards, label=f'Rewards  for {labels[idx]}')

        # Add labels and title
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.title('Rewards Over Episodes')

        # Add a legend
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Configuration parameters for the whole setup

    # Create the environment
    env_name = "LunarLander-v2"
    rander = False
    max_episode_steps=500
    number_of_episode=5000

    # Set seed for experiment reproducibility
    seed_ = 42
    # th.random.seed(seed_)
    np.random.seed(seed_)

    learning_rate = 0.001
    discount_factor = 0.99
    reword=[]
    test = [300]
    for hidden_layer in test:
        for i in range(1):
            print(f"i: {i} hidden_layer: {hidden_layer}")
            model = A2C(env_name,rander,max_episode_steps,number_of_episode, learning_rate, discount_factor, hidden_layer,seed_)
            reword.append(model.train())
    model.plot_resolt(reword,test)