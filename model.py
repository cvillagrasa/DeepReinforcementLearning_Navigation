from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BasicNetwork(nn.Module):
    state_size: int
    action_size: int
    n_hidden: int = 64

    def __post_init__(self):
        super().__init__()
        self.linear1 = nn.Linear(self.state_size, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)

    def forward(self, state):
        @try_various_attempts(2)
        def first_layer(state):
            return F.relu(self.linear1(state))

        x = first_layer(state)
        return F.relu(self.linear2(x))  # Second layer onwards work well if the first one already has returned output


@dataclass
class QNetwork(BasicNetwork):
    def __post_init__(self):
        super().__post_init__()
        self.linear3 = nn.Linear(self.n_hidden, self.action_size)

    def forward(self, state):
        x = super().forward(state)
        return self.linear3(x)


@dataclass
class DuelingQNetwork(BasicNetwork):
    def __post_init__(self):
        super().__post_init__()
        self.linear3_state = nn.Linear(self.n_hidden, 1)
        self.linear3_advantage = nn.Linear(self.n_hidden, self.action_size)

    def forward(self, state):
        x = super().forward(state)
        state_value = self.linear3_state(x)
        action_value = self.linear3_advantage(x)
        action_value -= action_value.mean(dim=-1, keepdim=True)
        return state_value + action_value


def try_various_attempts(allowed_attempts):
    """Dirty hack to prevent PyTorch 0.4 to halt at start on a RTX2070 Super with Cuda 9.0"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(allowed_attempts):
                try:
                    result = func(*args, **kwargs)
                    break
                except RuntimeError:  # When x.is_cuda, it usually just works on second attempt
                    if attempt + 1 == allowed_attempts:
                        raise RuntimeError(
                            f'RuntimeError: {allowed_attempts} successive failed attempts : ' +
                            f'Probably cublas runtime error'
                        )
                    else:
                        pass
            return result
        return wrapper
    return decorator
