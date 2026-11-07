import numpy as np


MOVES = ((1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2))


class KnightVsPawnsEnvironment:
  score_cap = 1200

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.knight = (3, 7)
    self.ghost = (5, 6)
    self.move_index = self.steps = self.frames = 0
    self.pressed = self.terminal = False
    self.pawns: list[tuple[int, int]] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    tap = pressed and not self.pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap: bool) -> int:
    if self.frames % 30 == 0:
      self._cycle_ghost()
    reward = 0
    if tap:
      self.knight = self.ghost
      self._cycle_ghost()
      reward = 8 - self.knight[1]
    if self.frames % 60 == 0:
      self.pawns = [(x, y + 1) for x, y in self.pawns if y < 7]
    if self.frames % 120 == 0:
      self.pawns.append((int(self.random.integers(8)), 0))
    self.frames += 1
    self.terminal = self.knight in self.pawns
    return reward

  def _cycle_ghost(self) -> None:
    x, y = self.knight
    for _ in MOVES:
      dx, dy = MOVES[self.move_index]
      self.move_index = (self.move_index + 1) % len(MOVES)
      if 0 <= x + dx < 8 and 0 <= y + dy < 8:
        self.ghost = (x + dx, y + dy)
        return

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    x, y = self.knight
    state[y + 1, x + 1, 0] = True
    x, y = self.ghost
    state[y + 1, x + 1, 1] = True
    for x, y in self.pawns:
      state[y + 1, x + 1, 2] = True
      if y < 7:
        state[y + 2, x + 1, 3] = True
    state[1:9, 1:9, 4] = True
    values = (self.steps, self.frames % 120, self.move_index)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
