from dataclasses import dataclass
from pathlib import Path
from collections import deque
import numpy as np
import torch
from unityagents import UnityEnvironment
from agent import DeepQLearningAgent


@dataclass
class BananaEnvironment:
    env_path = Path('../p1_navigation/Banana_Linux/Banana.x86_64')
    env = None
    action_space_size = None
    observation_space_size = None
    agent_params: dict = None  # Extra parameters for DeepQLearning are passed as a dictionary
    base_port: int = 5005
    train_mode: bool = False  # True for training / False for inference // removed in current version of ML-Agents

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load(self):
        self.env = UnityEnvironment(file_name=str(self.env_path), base_port=self.base_port)
        self.observation_space_size = self.get_observation_space_size()
        self.action_space_size = self.get_action_space_size()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent_params = {} if self.agent_params is None else self.agent_params
        self.agent = DeepQLearningAgent(state_size=self.observation_space_size, action_size=self.action_space_size,
                                        device=self.device, **self.agent_params)

    def close(self):
        self.env.close()
        self.env = None
        self.agent = None

    def reset(self):
        return self.env.reset(train_mode=self.train_mode)

    @property
    def default_brain_name(self):
        return self.env.brain_names[0]

    @property
    def default_brain(self):
        return self.env.brains[self.default_brain_name]

    def get_action_space_size(self):
        return self.default_brain.vector_action_space_size

    def get_observation_space_size(self):
        env_info = self.reset()
        vector_observations = env_info[self.default_brain_name].vector_observations
        return len(vector_observations[0])

    def select_random_action(self):
        return np.random.randint(self.action_space_size)

    def train(self, n_episodes=2000, max_t=10000, eps_start=1., eps_end=0.01, eps_decay=0.995,
                 metrics_log_size=100, save=False, saveas='checkpoint'):
        scores = []
        max_score = 0
        scores_window = deque(maxlen=metrics_log_size)
        loss_window = deque(maxlen=metrics_log_size)
        epsilon = eps_start
        for episode in range(n_episodes):
            score = self.train_episode(max_t, epsilon)
            scores_window.append(score)
            scores.append(score)
            if self.train_mode:
                loss_window.append(float(self.agent.loss))
                epsilon = max(eps_end, eps_decay * epsilon)
            self.print_episode_info(episode, scores_window, loss_window, epsilon, metrics_log_size)
            if self.train_mode and save and score > max_score:
                self.save_parameters(saveas)
            max_score = score if score > max_score else max_score
        return scores

    def train_episode(self, max_t, epsilon):
        env_info = self.reset()[self.default_brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            reward, next_state, done = self.train_step(state, epsilon)
            score += reward
            state = next_state
            if done:
                break
        return score

    def train_step(self, state, epsilon):
        action = self.agent.action(state, epsilon)
        env_info = self.env.step(action)[self.default_brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        if self.train_mode:
            self.agent.step(state, action, reward, next_state, done)
        return reward, next_state, done

    def play(self, n_episodes=2000, max_t=10000, metrics_log_size=1):
        self.train_mode = False
        return self.train(n_episodes=n_episodes, max_t=max_t, metrics_log_size=metrics_log_size, eps_start=0.)

    def print_episode_info(self, episode, scores_window, loss_window, epsilon, metrics_log_size):
        line = self.episode_line(episode, scores_window, loss_window, epsilon)
        print(line, end='')
        if episode % metrics_log_size == 0:
            print(line)

    def episode_line(self, episode, scores_window, loss_window, epsilon):
        episode += 1
        line = f'\rEpisode {episode}'
        line += f'\tAverage Score: {np.mean(scores_window):.2f}'
        if self.train_mode:
            line += f'\tEpsilon: {epsilon:.3f}'
            line += f'\tLoss: {np.mean(loss_window)}'
        return line

    def save_parameters(self, saveas='checkpoint'):
        savepath = Path() / 'checkpoints'
        savepath.mkdir(exist_ok=True)
        torch.save(self.agent.qnetwork_local.state_dict(), savepath / f'{saveas}.pt')

    def load_parameters(self, filename):
        unpickled_state_dict = torch.load(filename)
        self.agent.qnetwork_local.load_state_dict(unpickled_state_dict)
        self.agent.qnetwork_local.eval()
        self.train_mode = False
