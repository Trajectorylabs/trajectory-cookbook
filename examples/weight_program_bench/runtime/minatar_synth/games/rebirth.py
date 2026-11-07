from dataclasses import dataclass

import numpy as np


@dataclass
class Arrival:
  x: float
  world: int
  y: float = 37


class RebirthEnvironment:
  score_cap = 60

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y, self.offset = 110.0, 37.0, 10.0
    self.vx = self.vy = 0.0
    self.world = self.track_world = self.diamond_world = 1
    self.track_timer = self.steps = 0
    self.diamond_timer = 60
    self.diamond_count = self.multiplier = 1
    self.mode = "run"
    self.held = self.terminal = False
    self.tracks = []
    self.diamonds = []

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
    self.track_timer -= 1
    if self.track_timer < 0:
      self.tracks.append(Arrival(100 + self.track_world * 103, self.track_world))
      self.track_world *= -1
      self.track_timer = self.random.uniform(50, 60)
    for track in self.tracks:
      track.x -= track.world
    self.tracks = [track for track in self.tracks if (track.x - 100) * track.world >= 3]

    self.x += self.vx
    self.y += self.vy
    self.vx *= 0.792
    self.vy *= 0.99
    if self.mode == "run":
      self.x += (100 + self.world * self.offset - self.x) * 0.05
      self.offset = min(80, self.offset + 0.3)
      if tap:
        self.mode, self.vy = "jump", -2
    else:
      jumping = self.mode == "jump"
      if jumping:
        self.vy += 0.07 if self.held else 0.14
      if self.y > 37:
        self.y, self.vy, self.mode = 37, 0, "run"
      if jumping and tap:
        self.mode, self.vy = "land", 4
    if abs(self.vx) < 2 and any(self._touches(track) for track in self.tracks):
      self.vx = -self.world * self.offset * 0.2
      self.world *= -1
      self.offset = 10
      self.multiplier = max(1, self.multiplier - 1)

    self.diamond_timer -= 1
    if self.diamond_timer < 0:
      self.diamond_count -= 1
      if self.diamond_count < 0:
        self.diamond_world *= -1
        self.diamond_count = int(self.random.integers(4))
      y = 37 if self.random.random() < 0.5 else self.random.uniform(20, 30)
      self.diamonds.append(Arrival(100 + self.diamond_world * 103, self.diamond_world, y))
      self.diamond_timer = self.random.uniform(120, 150)
    reward = 0
    remaining = []
    for diamond in self.diamonds:
      diamond.x -= diamond.world / 3
      if self._touches(diamond):
        reward += self.multiplier
        self.multiplier += 1
      else:
        self.terminal |= (diamond.x - 100) * diamond.world < 3
        remaining.append(diamond)
    self.diamonds = remaining
    return reward

  def _touches(self, arrival):
    return abs(arrival.x - self.x) < 5 and abs(arrival.y - self.y) < 5

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 5, 0, 9)), int(np.clip(self.x // 20, 0, 9)), 0] = True
    for channel, arrivals in ((1, self.tracks), (2, self.diamonds)):
      for arrival in arrivals:
        if 0 <= arrival.x < 200:
          state[int(arrival.y // 5), int(arrival.x // 20), channel] = True
    state[8:, 5:, 3] = True
    state[:8, :5, 3] = True
    target = 100 + self.world * self.offset
    state[7, int(target // 20), 4] = True
    gauges = (self.steps, self.multiplier, ("run", "jump", "land").index(self.mode),
              self.held, round((self.vy + 2) * 50))
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
