"""A2C agent for Reacher task"""

import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unityagents import UnityEnvironment


class Config:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __int__(self):
        self.parser = argparse.ArgumentParser()
        self.history_length = 4
        self.num_workers = 20
        self.discount = 0.99
        self.use_gae = True
        self.gae_tau = 1.0
        self.entropy_weight = 0.01
        self.rollout_length = 5
        self.gradient_clip = 5
        self.max_steps = int(2e7)
        self.state_dim = 33
        self.action_dim = 4
        self.file_name = "<file_name>"
        self.logger = logging.getLogger('a2c_agent')
        self.eval_episodes = 2

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])


class Util:
    @staticmethod
    def tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
        return x

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    @staticmethod
    def random_seed():
        np.random.seed()
        torch.manual_seed(np.random.randint(int(1e6)))


class ActorCriticNet(nn.Module):

    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = Util.layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = Util.layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())


class GaussianActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = Util.tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, Util.tensor(np.zeros((log_prob.size(0), 1))), v


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [Util.layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class UnityEnvironmentTask:

    def __init__(self, file_name, train_mode=True):
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        self.action_dim = self.brain.vector_action_space_size
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_dim)

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(np.clip(actions, -1, 1))[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        terminals = env_info.local_done
        if np.any(terminals):
            next_states = self.reset()
        return next_states, rewards, terminals, env_info


class A2CAgent:
    def __init__(self, config):
        self.config = config
        self.task = UnityEnvironmentTask(file_name=config.file_name)
        self.network = GaussianActorCriticNet(config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim),
                                              critic_body=FCBody(config.state_dim))
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.0007)
        self.total_steps = 0
        self.states = self.task.reset()

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)

    def step(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network(states)
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
            self.online_rewards += rewards
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0

            rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
            states = next_states

        self.states = states
        pending_value = self.network(states)[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = Util.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, terminals, entropy = rollout[i]
            terminals = Util.tensor(terminals).unsqueeze(1)
            rewards = Util.tensor(rewards).unsqueeze(1)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_step(self, states):
        raise Exception('eval_step not implemented')

    def eval_episode(self):
        env = self.task = UnityEnvironmentTask(file_name=self.config.file_name, train_mode=False)
        states = env.reset()
        total_rewards = 0
        while True:
            action = self.eval_step(states)
            states, rewards, dones, _ = env.step(action)
            total_rewards += rewards
            if np.any(dones):
                break
        return total_rewards

    def eval_episodes(self):
        rewards = []
        for ep in range(self.config.eval_episodes):
            rewards.append(self.eval_episode())
        self.config.logger.info('evaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))


def run_steps(agent):
    Util.random_seed()
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            config.logger.info('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                agent.total_steps, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
