from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Body:
  x: float
  y: float
  vx: float = 0
  vy: float = 0

  def move(self, scroll: float = 0) -> None:
    self.x += self.vx - scroll
    self.y += self.vy


@dataclass
class Spike:
  x: float
  base: float
  height: float

  def hits(self, body: Body) -> bool:
    depth = (body.y - self.base) * (1 if self.height > 0 else -1)
    height = abs(self.height)
    width = 5 * (1 - max(0, depth - 3) / height)
    return -3 < depth < height + 3 and abs(body.x - self.x) < width + 3


class KiteEnvironment:
  score_cap = 3000

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.kite, self.player = Body(40, 50), Body(20, 87)
    self.winds: list[Body] = []
    self.coins: list[Body] = []
    self.spikes: list[Spike] = []
    self.spike_angles = [(0.0, 0.0), (math.pi / 2, math.pi)]
    self.spike_distance = self.coin_distance = self.wind_timer = 0.0
    self.steps = 0
    self.pressed = self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.pressed = action == 5
    self.steps += 1
    reward = 0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance(self) -> int:
    kite, player = self.kite, self.player
    scroll = max(0, kite.x - 60) * 0.1
    reward = self._move_spikes(scroll)
    kite.vx = (kite.vx + 0.2 * self.pressed) * 0.95
    kite.vy = (kite.vy + 0.01 - 0.2 * self.pressed) * 0.95
    kite.move(scroll)
    if not 0 <= kite.y <= 87:
      kite.y = min(87, max(0, kite.y))
      kite.vy = 0
    direction = 1 if kite.y < 50 else -1
    bumps = 0
    while bumps < 9 and any(spike.hits(kite) for spike in self.spikes):
      kite.y += 3 * direction
      bumps += 1
    if bumps:
      kite.vy = math.sqrt(bumps) * direction * 2
    player.move(scroll)
    if player.y < 87:
      player.vy += math.sqrt(99 - player.y) * 0.01
      player.vx *= 0.95
    else:
      player.y, player.vy = 87, 0
      player.vx *= 0.9
    if any(spike.hits(player) for spike in self.spikes):
      self.terminal = True
      return reward
    dx, dy = kite.x - player.x, kite.y - player.y
    distance = math.hypot(dx, dy)
    if distance > 50:
      pull = (distance - 50) / distance
      player.vx += dx * pull * 0.05
      player.vy += dy * pull * 0.05
      kite.vx -= dx * pull * 0.01
      kite.vy -= dy * pull * 0.01
      kite.x -= dx * pull
      kite.y -= dy * pull
    self.wind_timer -= 3 if self.pressed else 1
    if self.wind_timer < 0:
      speed = self.random.uniform(1, 2) * (2 if self.pressed else 1)
      self.winds.append(Body(-3, self.random.uniform(0, 87), vx=speed))
      self.wind_timer += 30
    surviving = []
    for wind in self.winds:
      wind.move()
      if abs(wind.x - kite.x) < 6 and abs(wind.y - kite.y) < 6:
        kite.vx += wind.vx
      elif wind.x <= 203:
        surviving.append(wind)
    self.winds = surviving
    return reward + self._move_coins(scroll)

  def _move_spikes(self, scroll: float) -> int:
    self.spike_distance -= scroll
    spawn = self.spike_distance < 0
    if spawn:
      self.spike_distance += 10
      for index, (a, b) in enumerate(self.spike_angles):
        height = math.sin(a) * 9 + math.sin(b) * 20 - 10
        self.spike_angles[index] = (a + self.random.uniform(0, 0.2), b + self.random.uniform(0, 0.4))
        if height > 0:
          self.spikes.append(Spike(205, 90 if index else 0, -height if index else height))
    for spike in self.spikes:
      spike.x -= scroll
    self.spikes = [spike for spike in self.spikes if spike.x >= -5]
    return int(spawn)

  def _move_coins(self, scroll: float) -> int:
    self.coin_distance -= scroll
    if self.coin_distance < 0:
      self.coins.append(Body(203, self.random.uniform(30, 80), vx=-0.5))
      self.coin_distance += self.random.uniform(199, 299)
    player = self.player
    reward = 0
    remaining = []
    for coin in self.coins:
      coin.move(scroll)
      if any(spike.hits(coin) for spike in self.spikes):
        continue
      dx, dy = player.x - coin.x, player.y - coin.y
      distance = math.hypot(dx, dy)
      if abs(dx) < 6 and abs(dy) < 6:
        reward += min(99, max(1, math.ceil(player.vx * 5))) * 50
        continue
      if any(abs(w.x - coin.x) < 6 and abs(w.y - coin.y) < 6 for w in self.winds):
        coin.x += 1
      if 0 < distance < 24:
        coin.vx = (coin.vx + dx / distance) * 0.9
        coin.vy = (coin.vy + dy / distance) * 0.9
      if -3 <= coin.x <= 207 and -3 <= coin.y <= 90:
        remaining.append(coin)
    self.coins = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for channel, objects in enumerate(([self.player], [self.kite], self.winds, self.coins)):
      for body in objects:
        if 0 <= body.x < 200 and 0 <= body.y < 100:
          state[int(body.y // 10), int(body.x // 20), channel] = True
    for spike in self.spikes:
      if 0 <= spike.x < 200:
        start, end = sorted((spike.base, spike.base + spike.height))
        state[int(start // 10):int(end // 10) + 1, int(spike.x // 20), 4] = True
    for row, value in enumerate((self.steps, round((self.kite.vy + 10) * 20),
                                 round((self.kite.vx + 5) * 20))):
      for bit in range(9):
        state[row, bit, 5] = bool(min(511, max(0, value)) & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
