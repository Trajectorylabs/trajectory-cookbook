from dataclasses import dataclass

import numpy as np


@dataclass
class Monster:
  x: float = 99
  y: float = 87
  vy: float = 0
  jumping: bool = False


class MonjumEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50.0, 87.0
    self.vx = self.vy = 0.0
    self.floor_timer = self.monster_timer = self.coin_timer = 0.0
    self.floors = [(0.0, 99.0)]
    self.monsters: list[Monster] = []
    self.coins: list[float] = []
    self.steps = self.frames = self.items = 0
    self.pressed = self.jumping = self.falling = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    release = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _ground(self, x):
    return any(start - 3 < x < start + width + 3 for start, width in self.floors)

  def _advance(self, release):
    self.floor_timer -= 0.5
    if self.floor_timer < 0:
      width = int(self.random.integers(50, 90))
      self.floors.append((110, width))
      self.floor_timer += 10 + width
    self.floors = [(x - 0.5, w) for x, w in self.floors if x - 0.5 + w > 0]
    self.monster_timer -= 1
    if self.monster_timer < 0:
      self.monsters.append(Monster())
      self.monster_timer += self.random.uniform(200, 300)
    for monster in self.monsters:
      monster.x -= 0.75
      monster.y += monster.vy
      if monster.jumping:
        monster.vy += 0.09
        if monster.y > 87:
          monster.y, monster.vy, monster.jumping = 87, 0, False
      elif not self._ground(monster.x - 4):
        monster.jumping, monster.vy = True, -1.4
      if abs(monster.x - self.x) < 5:
        self.terminal |= abs(monster.y - self.y) < 5
        self.items |= 2 if abs(monster.y - self.y) < 27 else 0
    self.monsters = [monster for monster in self.monsters if monster.x > -3]
    self.coin_timer -= 1
    if self.coin_timer < 0:
      self.coins.append(99)
      self.coin_timer += self.random.uniform(50, 100)
    remaining = []
    for x in self.coins:
      x -= 1
      if abs(x - self.x) < 5 and abs(self.y - 60) < 5:
        self.items |= 4
      elif x > -3:
        remaining.append(x)
    self.coins = remaining
    gap = not self._ground(self.x)
    if gap:
      self.items |= 1
    reward = 0
    if self.falling:
      self.vy += 0.14
      self.terminal |= self.y > 99
    elif self.jumping:
      self.vy += 0.12
      if self.y > 87:
        self.y, self.vy = 87, 0
        if gap:
          self.falling = True
        else:
          self.jumping = False
          reward = (0, 1, 5, 20)[self.items.bit_count()]
    else:
      self.falling = gap
      self.vx = -0.5 if self.pressed else 0
      if not self.pressed:
        self.x += (50 - self.x) * 0.02
      if release and self.frames > 5:
        self.vy, self.jumping, self.items = -2.5, True, 0
    self.x = min(50, max(0, self.x + self.vx))
    self.y += self.vy
    self.frames += 1
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, int(self.y // 10)), int(self.x // 10), 0] = True
    for x, width in self.floors:
      state[9, max(0, int(x // 10)):min(9, int((x + width) // 10)) + 1, 1] = True
    for monster in self.monsters:
      if 0 <= monster.x < 100:
        state[int(monster.y // 10), int(monster.x // 10), 2] = True
    for x in self.coins:
      if 0 <= x < 100:
        state[6, int(x // 10), 3] = True
    state[min(9, max(0, int((self.y + 4 * self.vy) // 10))), int(self.x // 10), 4] = True
    for row, value in enumerate((self.steps, self.items, round((self.vy + 3) * 60))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, :3, 5] = self.pressed, self.jumping, self.falling
    return state
