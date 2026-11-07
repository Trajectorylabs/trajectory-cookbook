from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Photon:
  x: float
  direction: int
  color: int
  reflected: bool = False


class PhotonLineEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.angle = self.turn_target = 0.0
    self.rotating = self.pressed = self.terminal = False
    self.hands = [[0], [1]]
    self.target = [[0], [1]]
    self.photons: list[Photon] = []
    self.world, self.stage, self.colors = 1, 0, 2
    self.steps = self.transition = 0
    self._next_stage()

  def _next_stage(self):
    if self.stage >= min(5, self.world + 1):
      self.world += 1
      self.stage = 0
      self.colors = min(4, int(self.random.integers(2, self.world + 2)))
      self.target = [[int(self.random.integers(self.colors))] for _ in range(2)]
      self.hands = [hand.copy() for hand in self.target]
    previous = [hand.copy() for hand in self.target]
    self.target = [hand[:4] for hand in self.target]
    ratio = 3 / self.colors
    additions = round(self.random.uniform(1, self.world) * ratio)
    removals = round(min(2, max(0, self.random.uniform(0, self.world) - 1)) * ratio)
    if additions == removals:
      removals -= 1
    for _ in range(removals):
      hand = self.target[self.random.integers(2)]
      if hand:
        hand.pop()
    for _ in range(additions + removals):
      self.target[self.random.integers(2)].append(int(self.random.integers(self.colors)))
    self.target = [hand[:5] for hand in self.target]
    if self.target == previous:
      hand = next((h for h in self.target if h), self.target[0])
      if hand:
        hand[-1] = (hand[-1] + 1) % self.colors
      else:
        hand.append(0)
    self.stage += 1
    self.timer, self.width = 60.0, 30.0
    self.bag = []

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
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

  def _advance(self, tap):
    reward = 0
    if self.transition:
      self.transition -= 1
      if self.width > -3:
        reward = self.world
        self.width -= 1
      if not self.transition:
        self._next_stage()
    else:
      self.timer -= 1
      if self.timer < 0:
        if not self.bag:
          self.bag = self.random.permutation(list(range(self.colors)) * 2
                                             + [4] * math.ceil(self.colors * 0.4)).tolist()
        direction = 1 if self.random.random() < 0.5 else -1
        self.photons.append(Photon(self.width if direction == 1 else 200 - self.width,
                                   direction, self.bag.pop()))
        self.timer = self.random.uniform(60, 90)
      self._move_photons()
    if tap and not self.rotating and not self.transition:
      self.rotating = True
      self.turn_target += math.pi
    if self.rotating:
      self.angle += 0.2
      if self.angle >= self.turn_target:
        if self.pressed:
          self.turn_target += math.pi
        else:
          self.angle = self.turn_target
          self.rotating = False
      if self.angle >= math.tau:
        self.angle -= math.tau
        self.turn_target -= math.tau
    self.width -= 0.02
    self.terminal = not self.transition and self.width < -3
    return reward

  def _move_photons(self):
    remaining = []
    swapped = math.pi / 2 <= self.angle <= 3 * math.pi / 2
    for photon in self.photons:
      direction = photon.direction
      photon.x += direction * (-2 if photon.reflected else 0.4)
      if photon.reflected:
        if self.width <= photon.x <= 200 - self.width:
          remaining.append(photon)
        continue
      hand = self.hands[(direction < 0) ^ swapped]
      edge = 100 - (7 * len(hand) + 6) * direction
      if (photon.x - edge) * direction <= 0:
        remaining.append(photon)
      elif self.rotating:
        photon.reflected = True
        remaining.append(photon)
      else:
        if photon.color == 4:
          if hand:
            hand.pop()
        else:
          hand.append(photon.color)
          if len(hand) > 9:
            self.width /= 2
        direct = self.hands == self.target
        reverse = self.hands[::-1] == self.target
        if direct or reverse:
          if (direct and swapped) or (not direct and reverse and not swapped):
            self.rotating = True
            self.turn_target += math.pi
          self.transition = 60
          self.photons = []
          return
    self.photons = remaining

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    for row, hand in enumerate((*self.target, *self.hands)):
      for col, color in enumerate(hand[:10]):
        state[row, col, color] = True
    for photon in self.photons:
      col = min(9, max(0, int(photon.x // 20)))
      row = 4 + 2 * (photon.direction < 0) + photon.reflected
      state[row, col, photon.color] = True
    values = (self.steps, round(self.angle * 80), self.world, self.stage,
              self.transition, round((self.width + 4) * 10))
    for channel, value in enumerate(values):
      for bit in range(9):
        state[8, bit, channel] = bool(value & (1 << bit))
    state[9, :2, 5] = self.pressed, self.rotating
    return state
