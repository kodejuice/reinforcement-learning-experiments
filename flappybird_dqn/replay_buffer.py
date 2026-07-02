import numpy as np


class ReplayBuffer:
  """
  An efficient, pre-allocated circular replay buffer for DQN.
  Stores transitions in contiguous numpy arrays for fast random sampling.
  """

  def __init__(self, capacity, state_dim):
    self.capacity = capacity
    self.state_dim = state_dim

    self.states = np.zeros((capacity, state_dim), dtype=np.float32)
    self.actions = np.zeros(capacity, dtype=np.int32)
    self.rewards = np.zeros(capacity, dtype=np.float32)
    self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
    self.dones = np.zeros(capacity, dtype=np.bool_)

    self.idx = 0
    self.size = 0

  def push(self, state, action, reward, next_state, done):
    """
    Store a new transition in the buffer.
    """
    self.states[self.idx] = state
    self.actions[self.idx] = action
    self.rewards[self.idx] = reward
    self.next_states[self.idx] = next_state
    self.dones[self.idx] = done

    self.idx = (self.idx + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size):
    """
    Randomly sample a batch of transitions.

    Returns:
        tuple: (states, actions, rewards, next_states, dones) as numpy arrays.
    """
    idxs = np.random.randint(0, self.size, size=batch_size)
    return (
        self.states[idxs],
        self.actions[idxs],
        self.rewards[idxs],
        self.next_states[idxs],
        self.dones[idxs]
    )

  def __len__(self):
    return self.size
