from dataclasses import dataclass

import numpy as np


@dataclass
class Target:
  x: float
  y: float
  timer: float = 0


@dataclass
class Bomb:
  x: float
  y: float
  vx: float = 2
  vy: float = 0


class ScrambirdEnvironment:
  score_cap = 100

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.y, self.vy, self.fuel = 50., 0., 50.
    self.height, self.slope, self.tank_timer = 10, 0, 10
    self.steps = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.terrain = [(i * 10., 10) for i in range(11)]
    self.shots, self.bombs, self.missiles, self.tanks = [], [], [], []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self, tap):
    for index, (x, height) in enumerate(self.terrain):
      x -= 0.3
      if x < -10:
        x += 110
        next_height = self.height + self.slope
        if not 10 <= next_height <= 50:
          self.slope *= -1
        else:
          self.height = next_height
          if self.random.random() < 0.2:
            self.slope = 0
          elif self.random.random() < 0.3:
            self.slope = self.random.choice((-10, 10))
        height = self.height
        self.tank_timer -= 1
        if self.tank_timer < 0:
          self.tanks.append(Target(x + 5, 87 - height))
          self.tank_timer = self.random.uniform(1, 16)
        elif self.random.random() < 0.5:
          timer = 9999 if self.random.random() < 0.5 else self.random.uniform(200, 300)
          self.missiles.append(Target(x + 5, 87 - height, timer))
      self.terrain[index] = x, height
    if tap:
      self.vy -= 0.5 if self.fuel > 0 else 0.1
      self.shots.append(Target(10, self.y))
      self.bombs.append(Bomb(10, self.y))
    self.vy = (self.vy + 0.015) * 0.98
    self.y += self.vy
    self.terminal = self._wall_hit(10, self.y, 3)
    for shot in self.shots:
      shot.x += 2
    self.shots = [shot for shot in self.shots if shot.x <= 103 and not self._wall_hit(shot.x, shot.y, 1)]
    for bomb in self.bombs:
      bomb.vx *= 0.9
      bomb.vy = (bomb.vy + 0.1) * 0.9
      bomb.x += bomb.vx
      bomb.y += bomb.vy
    self.bombs = [bomb for bomb in self.bombs if bomb.y <= 103 and not self._wall_hit(bomb.x, bomb.y, 1)]
    reward = 0
    spent = set()
    for targets, fuel_tanks in ((self.missiles, False), (self.tanks, True)):
      survivors = []
      for target in targets:
        target.x -= 0.3
        if not fuel_tanks:
          target.timer -= 1
          if target.timer < 0:
            target.y -= 0.5
        hits = [projectile for projectile in (*self.shots, *self.bombs)
                if abs(projectile.x - target.x) < 4 and abs(projectile.y - target.y) < 4]
        if hits:
          spent.update(map(id, hits))
          if fuel_tanks:
            self.fuel = min(50, self.fuel + 10)
          else:
            reward += self.multiplier
            self.multiplier += 1
          continue
        self.terminal |= abs(target.x - 10) < 5 and abs(target.y - self.y) < 5
        if target.x >= -3 and target.y >= -3:
          survivors.append(target)
        elif not fuel_tanks:
          self.multiplier = max(1, self.multiplier - 1)
      targets[:] = survivors
    self.shots = [shot for shot in self.shots if id(shot) not in spent]
    self.bombs = [bomb for bomb in self.bombs if id(bomb) not in spent]
    self.fuel = max(0, self.fuel - 0.025)
    return reward

  def _wall_hit(self, x, y, margin):
    return y - margin < 5 or any(
      left - margin < x < left + 9 + margin and y + margin > 90 - height
      and y - margin < 90 for left, height in self.terrain
    )

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), 1, 0] = True
    state[0, :, 1] = True
    for col in range(10):
      for x, height in self.terrain:
        if x < col * 10 + 10 and x + 9 > col * 10:
          state[(90 - height) // 10:9, col, 1] = True
    for channel, entities in ((2, self.missiles), (3, self.tanks), (4, (*self.shots, *self.bombs))):
      for entity in entities:
        if 0 <= entity.x < 100 and 0 <= entity.y < 100:
          state[int(entity.y // 10), int(entity.x // 10), channel] = True
    values = (self.steps, round(self.fuel * 10), self.multiplier, self.held, round((self.vy + 30) * 16))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
