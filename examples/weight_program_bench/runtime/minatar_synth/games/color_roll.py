import numpy as np


HORIZON = 400
CENTER = 5
WIDTH = 10


class ColorRollEnvironment:
  score_cap = HORIZON * 2 ** 18

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = 0
    self.last_action = 0
    self.pressure = 0
    self.penalty = 1
    self.base = 0
    self.target_count = 2
    self.chain = 0
    self.shot_color: int | None = None
    self.shot_delay = 0
    self.terminal = False
    self.lanes: list[tuple[np.ndarray, int]] = []
    self.previous_palette: list[int] = []
    self._refill()

  def _refill(self) -> None:
    while len(self.lanes) < self.target_count:
      palette = self.previous_palette.copy()
      if not palette:
        palette.append(int(self.random.integers(4)))
      if len(palette) == 1 or (len(palette) < 4 and self.random.random() < 0.5):
        palette.append(int(self.random.integers(4)))
      else:
        palette.pop(int(self.random.integers(len(palette))))
      self.previous_palette = palette
      colors = np.resize(np.repeat(palette, (WIDTH + len(palette) - 1) // len(palette)), WIDTH)
      colors = np.roll(colors, int(self.random.integers(WIDTH)))
      self.lanes.append((colors, int(self.random.choice((-1, 1)))))

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    self.pressure += 1
    self.lanes = [(np.roll(colors, velocity), velocity) for colors, velocity in self.lanes]
    reward = 0
    if self.shot_delay:
      self.shot_delay -= 1
      if self.shot_delay == 0:
        reward = self._contact()
    elif pressed:
      self.shot_color = None
      self.chain = 0
      self.shot_delay = 2
      self.pressure += 4 * self.penalty
    if not self.shot_delay:
      self.terminal = self.pressure + 4 * len(self.lanes) >= 48
      if not self.terminal:
        self._refill()
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _contact(self) -> int:
    color = int(self.lanes[0][0][CENTER])
    if self.shot_color is not None and color != self.shot_color:
      self.penalty = max(1, min(4, self.penalty * (3 // (2 ** self.chain))))
      self.shot_color = None
      return 0
    self.shot_color = color
    self.lanes.pop(0)
    self.pressure = max(0, self.pressure - 2 ** self.chain)
    reward = 2 ** (self.base + self.chain)
    self.chain += 1
    if self.lanes:
      self.shot_delay = 1
    else:
      self.shot_color = None
      self.target_count += 1
      if self.target_count > min(10, 5 + self.base):
        self.base = min(9, self.base + 1)
        self.target_count = 2
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for row, (colors, velocity) in enumerate(reversed(self.lanes), 10 - len(self.lanes)):
      state[row, np.arange(WIDTH), colors] = True
      state[row, 0, 4] = velocity == 1
    values = (self.pressure, self.penalty, self.base, self.target_count,
              self.chain, self.shot_delay, 0 if self.shot_color is None else self.shot_color + 1,
              int(self.last_action == 5), self.steps)
    for row, value in enumerate(values):
      for bit in range(WIDTH):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
