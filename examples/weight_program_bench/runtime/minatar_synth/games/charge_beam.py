from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Target:
  x: float
  size: int = 1
  coin: bool = False
  suction: float = 0


class ChargeBeamEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.power = 0
    self.pressed = self.terminal = self.last_coin = False
    self.charge = self.penalty = 0.0
    self.spawn_distance = 30.0
    self.shot: float | None = None
    self.coin_combo = self.enemy_combo = self.penalty_combo = 1
    self.targets = [Target(150)] + [Target(x, coin=True) for x in (160, 170, 180)]

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
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    if self.shot is None:
      if self.pressed and self.charge < 99:
        self.charge += 1.5
      elif self.charge:
        self.power = 1 + math.ceil(max(0, self.charge - 25) / 9)
        self.shot, self.charge = 10.0, 0.0
        self.coin_combo = self.enemy_combo = self.penalty_combo = 1
    if self.shot is not None:
      self.shot += 2.5
      if self.shot > 203:
        self.shot = None

    self.penalty = max(0, self.penalty - 0.02)
    speed = (1 + self.penalty) * 0.5
    self.spawn_distance -= speed
    if self.spawn_distance < 0:
      self._spawn_wave()

    for target in self.targets:
      target.x -= target.suction or speed
      if target.suction:
        target.suction += 0.1
    first_enemy = min((target.x for target in self.targets if not target.coin), default=math.inf)

    reward = 0
    survivors = []
    for target in self.targets:
      if self._shot_hits(target):
        spent = self.power == 1
        if target.coin:
          reward -= self.penalty_combo
          self.penalty_combo = min(64, self.penalty_combo * 2)
          self.penalty += 0.5
          if spent:
            self.shot = None
          continue
        if target.size <= self.power:
          reward += target.size * self.enemy_combo
          self.enemy_combo += 1
          if target.size > 1:
            self.power -= target.size
          if spent or not self.power:
            self.shot = None
          continue
        target.size -= self.power
        self.shot = None
      if target.coin:
        if target.suction and target.x < 10:
          reward += self.coin_combo
          self.coin_combo = min(64, self.coin_combo * 2)
          continue
        if not target.suction and target.x < first_enemy:
          target.suction = 1
      elif target.x < 15:
        self.terminal = True
      survivors.append(target)
    self.targets = survivors
    if not any(target.x <= 200 and not target.suction for target in self.targets):
      self.spawn_distance = 0
    return reward

  def _shot_hits(self, target: Target) -> bool:
    return self.shot is not None and (
      self.shot - 5 <= target.x + 3 * (target.size - 1)
      and target.x - 5 <= self.shot + 3 * (self.power - 1)
    )

  def _spawn_wave(self) -> None:
    self.last_coin = not self.last_coin and self.random.random() < 0.5
    x = 200
    for _ in range(int(self.random.integers(3, 9))):
      size = 1
      if not self.last_coin and self.random.random() < 0.3:
        size = int(self.random.integers(2, 6))
      self.targets.append(Target(x, size, self.last_coin))
      x += 10 + math.ceil((size - 1) / 2) * 6
    self.spawn_distance = x - 190

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 0, 0] = True
    for target in self.targets:
      if 0 <= target.x < 200:
        col = int(target.x) // 20
        if target.coin:
          state[5, col, 3 if target.suction else 2] = True
        else:
          state[6 - target.size:6, col, 1] = True
    if self.shot is not None and self.shot < 200:
      start = int(self.shot) // 20
      end = int(self.shot + 3 * (self.power - 1)) // 20
      state[5, start:end + 1, 4] = True
    for row, value in enumerate((self.steps, round(self.charge), self.power, round(self.penalty * 10),
                                 self.coin_combo, self.enemy_combo, self.penalty_combo)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = self.pressed
    return state
