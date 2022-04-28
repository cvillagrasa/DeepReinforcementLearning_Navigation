from dataclasses import dataclass
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork, DuelingQNetwork


@dataclass
class DeepQLearningAgent:
    state_size: ...
    action_size: ...
    buffer_size: ... = int(1e5)  # replay buffer size
    batch_size: ... = 64  # minibatch size
    gamma: ... = 0.99  # discount factor
    tau: ... = 1e-3  # for soft update of target parameters
    lr: ... = 5e-4  # learning rate
    update_every: ... = 4  # how often to update the network
    ddqn: ... = False  # use of Double Deep Q Learning algorithm
    prioritized_experience_replay: ... = 0.  # 0 -> pure random sampling, 1 -> full prioritized experience replay
    dueling: ... = False  # use of Dueling Deep Q Learning algorithm
    e: ... = 1e-4  # constant to prevent samples from being starved in prioritized experience replay
    beta: ... = 0.5  # non-uniform probabilities compensation in importance sampling for prioritized experience replay
    device: ... = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __post_init__(self):
        network = DuelingQNetwork if self.dueling else QNetwork
        self.qnetwork_local = network(self.state_size, self.action_size).to(self.device)
        self.qnetwork_target = network(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.loss = None
        self.memory_replay = MemoryReplayBuffer(self.action_size, self.buffer_size, self.batch_size,
                                                a=self.prioritized_experience_replay, beta=self.beta,
                                                device=self.device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1
        if self.memory_replay.prioritized:
            experience = self.memory_replay.experience(state, action, reward, next_state, done, 1)
            experience = self.memory_replay.experiences_to_tensors([experience])
            state, action, reward, next_state, done, priority = experience
            pred_qvalues = self.predicted_qvalues(state, action)
            computed_qvalues = self.computed_qvalues(reward, next_state, done)
            td_error = self.temporal_difference_error(pred_qvalues, computed_qvalues)
            priority = td_error.abs().item() + self.e
        else:
            priority = 1.
        self.memory_replay.add(state, action, reward, next_state, done, priority)
        if self.t_step % self.update_every == 0 and len(self.memory_replay) > self.batch_size:
            sample = self.memory_replay.sample()
            self.learn(sample['experiences'], sample['max_weight'])

    def get_action_values(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return action_values

    def action_epsilon_greedy(self, state, epsilon=0.):
        if random.random() > epsilon:
            action_values = self.get_action_values(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def action(self, state, epsilon=0.):
        return self.action_epsilon_greedy(state, epsilon=epsilon)

    def learn(self, experiences, max_weight):
        states, actions, rewards, next_states, dones, priorities = experiences
        pred_qvalues = self.predicted_qvalues(states, actions)
        computed_qvalues = self.computed_qvalues(rewards, next_states, dones)
        if self.memory_replay.prioritized:
            weight = ((len(self.memory_replay) * priorities) ** -self.beta / max_weight).squeeze()
            pred_qvalues *= weight
            computed_qvalues *= weight
        self.loss = F.mse_loss(pred_qvalues, computed_qvalues)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.soft_update()

    def predicted_qvalues(self, states, actions):
        pred_actions = self.qnetwork_local(states)
        return pred_actions.gather(1, actions).squeeze()

    def computed_qvalues(self, rewards, next_states, dones):
        if self.ddqn:
            best_actions_next = self.qnetwork_local(next_states).detach().argmax(dim=-1).unsqueeze(-1)
            qvalues_next = self.qnetwork_target(next_states).detach().gather(1, best_actions_next).squeeze()
        else:
            qvalues_next = self.qnetwork_target(next_states).detach().max(dim=-1)[0]
        return rewards.squeeze() + self.gamma * qvalues_next * (1 - dones.squeeze())

    def temporal_difference_error(self, pred_qvalues, computed_qvalues):
        return computed_qvalues - pred_qvalues

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)


@dataclass
class MemoryReplayBuffer:
    action_size: ...
    buffer_size: ...
    batch_size: ...
    device: ... = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a: ... = 0.  # 0 -> pure random sampling, 1 -> full prioritized experience replay
    beta: ... = 0.5  # non-uniform probabilities compensation in importance sampling for prioritized experience replay

    def __post_init__(self):
        self.buffer = deque(maxlen=self.buffer_size)
        self.field_names = ['state', 'action', 'reward', 'next_state', 'done', 'priority']
        self.experience = namedtuple('Experience', field_names=self.field_names)

    def add(self, state, action, reward, next_state, done, priority=1.):
        e = self.experience(state, action, reward, next_state, done, priority)
        self.buffer.append(e)

    def sample(self):
        if self.prioritized:
            denominator = self.sum_priorities
            probabilities = [e.priority ** self.a / denominator for e in self.buffer]
            sample_indexes = np.random.choice(np.arange(len(self.buffer)), size=self.batch_size, p=probabilities)
            experiences = [self.buffer[idx] for idx in sample_indexes]
            max_weight = (np.min(probabilities) * len(self)) ** -self.beta
        else:
            experiences = random.sample(self.buffer, self.batch_size)
            max_weight = 1.
        return {'experiences': self.experiences_to_tensors(experiences), 'max_weight': max_weight}

    def experiences_to_tensors(self, experiences):
        experiences = [[e[idx] for e in experiences if e is not None] for idx, field in enumerate(self.field_names)]
        experiences = [np.vstack(e) for e in experiences]
        experiences[4] = experiences[4].astype(np.uint8)
        experiences = [torch.from_numpy(e) for e in experiences]
        experiences = [e.long() if idx == 1 else e.float() for idx, e in enumerate(experiences)]
        return [e.to(self.device) for e in experiences]

    def __len__(self):
        return len(self.buffer)

    @property
    def prioritized(self):
        return self.a > 0

    @property
    def sum_priorities(self):
        return np.sum([e.priority ** self.a for e in self.buffer])
