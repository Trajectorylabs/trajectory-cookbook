import numpy as np


class ISlashEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.knockback, self.slash_distance, self.hit_x = 10.0, 0.0, 0.0, 0.0
    self.spawn_distance = 0.0
    self.steps = self.slash_timer = 0
    self.multiplier = 1
    self.ready = self.pressed = self.terminal = False
    self.enemies: list[float] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    self.ready |= self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    scroll = (self.x - 30) * 0.05
    self.spawn_distance -= scroll + (self.slash_timer == 0)
    if not self.enemies or self.spawn_distance <= 0:
      spacing = self.random.uniform(15, 30)
      count = self.random.integers(3, 10)
      self.enemies.extend(203 + i * spacing for i in range(count))
      self.spawn_distance = spacing * count + self.random.uniform(0, 30)
    self.x -= scroll + self.knockback
    if self.x < 0:
      self.terminal = True
      return 0
    self.knockback *= 0.9
    if self.knockback > 1:
      self.hit_x -= scroll
    slashing = self.ready and self.pressed
    reward = 0
    if slashing and self.knockback < 1 and not self.slash_timer:
      self.slash_distance = (self.enemies[0] - self.x) * 2
      reward += self._slash()
    if not self.slash_timer:
      if self.enemies and self.enemies[0] < self.x:
        self._hit(0)
    else:
      self.slash_timer -= 1
      if not self.slash_timer and slashing and self.enemies:
        reward += self._slash()
    drift = scroll + (self.slash_timer == 0)
    self.enemies = [x - drift for x in self.enemies]
    return reward

  def _slash(self) -> int:
    self.x += self.slash_distance
    crossed = self.enemies[0] < self.x
    overshot = len(self.enemies) > 1 and self.enemies[1] < self.x
    if overshot:
      self._hit(1)
    else:
      self.slash_timer = 10 if crossed else 0
    if crossed:
      self.enemies.pop(0)
      if not overshot:
        reward = self.multiplier
        self.multiplier += 1
        return reward
    return 0

  def _hit(self, index: int) -> None:
    self.hit_x = self.enemies[index]
    self.knockback = (self.x - self.hit_x + 9) * 0.25
    self.x = self.hit_x - 6
    self.slash_timer = 0
    self.multiplier = max(1, self.multiplier - 1)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    col = min(9, max(0, int(self.x // 20)))
    state[7, col, 0] = True
    for x in self.enemies:
      if 0 <= x < 200:
        state[7, int(x // 20), 1] = True
    if self.slash_timer:
      left = max(0, min(9, int((self.x - self.slash_distance) // 20)))
      state[7, min(left, col):max(left, col) + 1, 2] = True
    if self.knockback > 1:
      hit = min(9, max(0, int(self.hit_x // 20)))
      state[7, min(hit, col):max(hit, col) + 1, 3] = True
    state[8:, :, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.slash_timer,
                                 min(511, round(self.knockback * 10)))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.ready
    return state
