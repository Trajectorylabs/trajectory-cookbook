import numpy as np


TRACK_LENGTH = 16
WAVES = 4


class PakuPakuEnvironment:
  # Each wave can raise the multiplier once through a ghost capture.
  score_cap = float(sum(16 * (3 + 2 * wave) + 10 * (2 + 2 * wave) for wave in range(WAVES)))

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 500):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = horizon
    self.random = np.random.default_rng(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = 6
    self.ghost = 15
    self.direction = 1
    self.eyes = 0
    self.power = 0
    self.steps = 0
    self.wave = 0
    self.multiplier = 1
    self.last_action = 0
    self.terminal = False
    self._refill()

  def _refill(self) -> None:
    self.pellets = set(range(TRACK_LENGTH))
    self.power_cell = int(self.random.integers(1, 6) if self.player >= 8 else self.random.integers(10, 15))
    self.multiplier += 1

  @classmethod
  def normalize_score(cls, raw_score: float) -> float:
    return min(max(raw_score, 0.0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    if action == 5 and self.last_action != 5:
      self.direction *= -1
    elif action in (1, 3):
      self.direction = -1 if action == 1 else 1
    self.last_action = action
    self.steps += 1
    previous_player = self.player
    previous_ghost = self.ghost
    self.player = (self.player + self.direction) % TRACK_LENGTH
    reward = 0
    if self.player in self.pellets:
      self.pellets.remove(self.player)
      reward += self.multiplier
      if self.player == self.power_cell and not self.eyes:
        self.power = 16
    returning = bool(self.eyes)
    if returning:
      self.ghost += self.eyes
      if self.ghost in (0, TRACK_LENGTH - 1):
        self.eyes = 0
    else:
      interval = 3 if self.power else max(1, 3 - self.wave)
      if self.steps % interval == 0:
        direction = 1 if self.player > self.ghost else -1
        if self.power:
          direction *= -1
        self.ghost = max(0, min(TRACK_LENGTH - 1, self.ghost + direction))
    crossed = self.player == previous_ghost and self.ghost == previous_player
    if not returning and (self.player == self.ghost or crossed):
      if self.power:
        reward += 10 * self.multiplier
        self.multiplier += 1
        self.power = 0
        self.eyes = -1 if self.ghost >= 8 else 1
      else:
        self.terminal = True
    self.power = max(0, self.power - 1)
    if not self.terminal and not self.pellets:
      self.wave += 1
      if self.wave == WAVES:
        self.terminal = True
      else:
        self.power = 0
        self._refill()
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    observation = np.zeros((10, 10, 6), dtype=bool)
    observation[self.player // 8, self.player % 8, 0] = True
    ghost_channel = 3 if self.eyes or self.power else 1
    observation[self.ghost // 8, self.ghost % 8, ghost_channel] = True
    for pellet in self.pellets:
      observation[pellet // 8, pellet % 8, 3 if pellet == self.power_cell else 2] = True
    next_cell = (self.player + self.direction) % TRACK_LENGTH
    observation[next_cell // 8, next_cell % 8, 4] = True
    for row, value in enumerate((self.power, self.wave, self.multiplier, self.steps, self.eyes + 1, self.last_action), start=2):
      for bit in range(10):
        observation[row, bit, 5] = bool(value & (1 << bit))
    return observation
