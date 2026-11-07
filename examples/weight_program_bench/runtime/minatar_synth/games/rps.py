from dataclasses import dataclass

import numpy as np


@dataclass
class Hand:
  lane: int
  kind: int
  y: float = -9
  speed: float = 0.1


class RpsEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.lane = self.kind = self.freeze = self.steps = 0
    self.target_y = self.y = 97.0
    self.x, self.vy = 20.0, -1.0
    self.multiplier = 1
    self.held = self.terminal = False
    self.hands = []
    self.kinds = self.random.integers(3, size=4)
    self.timers = self.random.uniform(0, 99, size=4)

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
    self.timers -= 1
    for lane in np.flatnonzero(self.timers < 0):
      self.hands.append(Hand(lane, self.kinds[lane]))
      self.timers[lane] = self.random.uniform(200, 300)
      if self.random.random() < 0.4:
        self.kinds[lane] = self.random.integers(3)
    for hand in self.hands:
      hand.y += hand.speed
      hand.speed += ((0.2 if hand.y > 90 else 0.1) - hand.speed) * 0.1
      for other in self.hands:
        if other is not hand and other.lane == hand.lane and abs(other.y - hand.y) < 6:
          center = (other.y + hand.y) / 2
          hand.speed, other.speed = (0.4 * other.speed - 0.2 * hand.speed,
                                     0.4 * hand.speed - 0.2 * other.speed)
          offset = 3 if other.y < hand.y else -3
          hand.y, other.y = center + offset, center - offset
      self.terminal |= hand.y > 99
    self.freeze -= 1
    if self.freeze < 0 and tap:
      self.lane = (self.lane + 1) % 4
      self.kind = (self.kind + 1) % 3
      self.target_y, self.vy = 97, -1
    self.vy += ((-1 if self.freeze < 0 else 0) - self.vy) * 0.05
    self.target_y = min(97, max(10, self.target_y + self.vy))
    offset = 20 + self.lane * 20 - self.x
    self.x += offset if offset < 1 else offset * 0.5
    self.y += (self.target_y - self.y) * 0.5
    reward = 0
    remaining = []
    for hand in self.hands:
      if hand.lane == self.lane and abs(self.target_y - hand.y) < 5:
        outcome = (self.kind - hand.kind) % 3
        if outcome == 0:
          self.vy = 0
          hand.speed -= 4
          hand.y = self.target_y - 3
        elif outcome == 1:
          reward += self.multiplier
          self.multiplier += 1
          continue
        else:
          self.multiplier = 1
          self.vy = 5
          hand.speed -= 0.5
          self.target_y = hand.y + 3
          self.freeze = 60
      remaining.append(hand)
    self.hands = remaining
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, int(self.y // 10)), int(self.x // 10), 0] = True
    for hand in self.hands:
      if 0 <= hand.y < 100:
        state[int(hand.y // 10), 2 + 2 * hand.lane, 1 + hand.kind] = True
    state[min(9, int(self.target_y // 10)), 2 + 2 * self.lane, 4] = True
    values = (self.steps, self.kind, max(0, self.freeze), self.multiplier, self.held)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
