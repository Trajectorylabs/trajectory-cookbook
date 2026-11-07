from dataclasses import dataclass

import numpy as np


LANES = (22, 42, 62, 82)


@dataclass
class Target:
  x: float
  lane: int


class SLanesEnvironment:
  score_cap = 150

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.target_y = 3.0, 22.0, 22.0
    self.lane = self.steps = self.lane_timer = self.shot_timer = self.spawn_timer = 0
    self.multiplier = 1
    self.held = self.terminal = False
    self.enemies, self.coins, self.shots = [], [], []
    self.speeds = np.ones(4)

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
    self.spawn_timer -= 1
    if self.spawn_timer < 0:
      self.enemies.append(Target(103, int(self.random.integers(4))))
      self.spawn_timer = self.random.uniform(60, 90)
    if self.held:
      self.x = 3 if tap else min(50, self.x + 0.3)
      self.y = self.target_y = LANES[self.lane]
      self.shot_timer -= 1
      if self.shot_timer < 0:
        self.shots.append(Target(self.x, self.lane))
        self.shot_timer = self.lane_timer = 20
    else:
      self.lane_timer -= 1
      if self.lane_timer < 0:
        self.lane = (self.lane + 1) % 4
        self.target_y = LANES[self.lane] + (100 if self.lane == 0 else 0)
        self.y %= 100
        self.lane_timer = 20
      self.x += (3 - self.x) * 0.3
      self.y += (self.target_y - self.y) * 0.3
    self.speeds += (0.9 - self.speeds) * 0.02
    reward = 0
    coins = []
    for coin in self.coins:
      coin.x -= 0.2
      if abs(coin.x - self.x) < 5.5 and abs(LANES[coin.lane] - self.y % 100) < 4.5:
        reward += self.multiplier
        self.multiplier += 1
      elif coin.x < -3:
        self.speeds[coin.lane] += 1
        self.multiplier = max(1, self.multiplier - 1)
      else:
        coins.append(coin)
    for shot in self.shots:
      shot.x += 2
    enemies = []
    for enemy in self.enemies:
      enemy.x -= 0.2 * self.speeds[enemy.lane]
      if any(shot.lane == enemy.lane and abs(shot.x - enemy.x) < 6 for shot in self.shots):
        coins.append(enemy)
      else:
        self.terminal |= enemy.x < 9
        enemies.append(enemy)
    self.coins, self.enemies = coins, enemies
    self.shots = [
      shot for shot in self.shots if shot.x <= 103 and not any(
        shot.lane == enemy.lane and abs(shot.x - enemy.x) < 6 for enemy in enemies
      )
    ]
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(self.y % 100 // 10), int(self.x // 10), 0] = True
    for channel, entities in ((1, self.enemies), (2, self.coins), (3, self.shots)):
      for entity in entities:
        if 0 <= entity.x < 100:
          state[LANES[entity.lane] // 10, int(entity.x // 10), channel] = True
    state[LANES[self.lane] // 10, :, 4] = True
    gauges = (self.steps, self.multiplier, self.lane_timer, self.shot_timer,
              self.held, *(round(speed * 100) for speed in self.speeds))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
