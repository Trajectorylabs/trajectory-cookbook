from dataclasses import dataclass

import numpy as np


@dataclass
class Voice:
  x: float
  y: float
  vx: float
  size: float

  def move(self, edge):
    self.x += self.vx
    if not edge <= self.x <= 100 - edge:
      self.x = min(100 - edge, max(edge, self.x))
      self.vx *= -1


@dataclass
class Falling:
  x: float
  y: float
  speed: float
  size: float = 5

  def touches(self, voice):
    reach = (self.size + voice.size) / 2
    return abs(self.x - voice.x) < reach and abs(self.y - voice.y) < reach


class StillfireChoirEnvironment:
  score_cap = 400

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.player = Voice(50, 88, 0.72, 7)
    self.allies, self.enemies, self.items = [], [], []
    self.beams = []
    self.volley, self.enemy_timer, self.item_timer = 62, 25, 75
    self.steps = self.glow = 0
    self.held = self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0., True
    if self.random.random() >= self.sticky_action_prob:
      self.held = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self):
    voices = [self.player, *self.allies]
    if not self.held:
      self.player.move(5)
      for ally in self.allies:
        ally.move(4)
    self.volley -= 1
    firing = self.volley <= 0
    if firing:
      self.volley = 62
      self.glow = 4
    self.beams = [(voice.x, voice.y) for voice in voices] if self.glow else []
    self.glow = max(0, self.glow - 1)
    self.enemy_timer -= 1
    if self.enemy_timer <= 0:
      self.enemies.append(Falling(self.random.uniform(5, 95), -4,
                                  self.random.uniform(0.45, 0.78), self.random.uniform(5, 9)))
      self.enemy_timer = self.random.uniform(31, 49)
    self.item_timer -= 1
    if self.item_timer <= 0:
      self.items.append(Falling(self.random.uniform(7, 93), -3, self.random.uniform(0.36, 0.52)))
      self.item_timer = self.random.uniform(125, 175)
    reward = 0
    remaining = []
    for item in self.items:
      item.y += item.speed
      if any(item.touches(voice) for voice in voices):
        speed = self.random.uniform(0.48, 0.9) * self.random.choice((-1, 1))
        self.allies.append(Voice(item.x, item.y, speed, 5))
        reward += 5
      elif item.y <= 105:
        remaining.append(item)
    self.items = remaining
    remaining = []
    for enemy in self.enemies:
      enemy.y += enemy.speed
      half = enemy.size / 2
      if firing and any(abs(enemy.x - x) < half + 1 and
                        enemy.y + half > 6 and enemy.y - half < y - 2
                        for x, y in self.beams):
        reward += 10 + 2 * len(self.allies)
      else:
        self.terminal |= any(enemy.touches(voice) for voice in voices)
        if enemy.y <= 105:
          remaining.append(enemy)
    self.enemies = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    groups = ((self.player,), self.allies, self.enemies, self.items)
    for channel, entities in enumerate(groups):
      for entity in entities:
        if 0 <= entity.y < 100:
          state[int(entity.y // 10), int(entity.x // 10), channel] = True
    for x, y in self.beams:
      state[:int(y // 10) + 1, int(x // 10), 4] = True
    values = (self.steps, self.volley, self.held, len(self.allies), self.player.vx > 0, self.glow)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
