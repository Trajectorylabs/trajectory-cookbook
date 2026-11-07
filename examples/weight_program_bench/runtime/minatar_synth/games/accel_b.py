from dataclasses import dataclass

import numpy as np


HORIZON = 400
MIN_SPAWN_STEPS = 4
MAX_MULTIPLIER = 64
PLAYER_ROW = 4


@dataclass
class Target:
  row: int
  column: int
  fire_in: int = 0


@dataclass
class HomingShot:
  row: int
  column: int
  target: Target
  age: int = 0


class AccelBEnvironment:
  score_cap = (HORIZON // MIN_SPAWN_STEPS) * 2 * MAX_MULTIPLIER

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.throttle = 0
    self.spawn_in = MIN_SPAWN_STEPS
    self.multiplier = 1
    self.enemies: list[Target] = []
    self.hostile: list[Target] = []
    self.shots: list[HomingShot] = []
    self.blasts: list[tuple[int, int, int]] = []
    self.terminal = False

  @property
  def column(self) -> int:
    return 1 + self.throttle // 2

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.steps += 1
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.throttle = min(6, self.throttle + 1) if action == 5 else max(0, self.throttle - 1)
    if pressed and not (self.shots or self.blasts):
      self.multiplier = 1
      self.shots = [HomingShot(PLAYER_ROW, self.column, target)
                    for target in (*self.enemies, *self.hostile)]
    self._advance_targets()
    self._advance_salvo()
    reward = 0
    for targets in (self.enemies, self.hostile):
      remaining = []
      for target in targets:
        if any(abs(target.row - row) <= 1 and abs(target.column - column) <= 1
               for row, column, _ in self.blasts):
          reward += self.multiplier
          self.multiplier = min(MAX_MULTIPLIER, 2 * self.multiplier)
        elif target.column >= 0:
          remaining.append(target)
      targets[:] = remaining
    self.terminal = any((missile.row, missile.column) == (PLAYER_ROW, self.column)
                        for missile in self.hostile) or self.steps >= self.horizon
    self.spawn_in -= 1
    if self.spawn_in == 0 and not self.terminal:
      self.enemies.append(Target(int(self.random.choice((1, 2, 3, 5, 6, 7, 8))), 9, 3))
      self.spawn_in = max(MIN_SPAWN_STEPS, 10 - self.throttle)
    return float(reward), self.terminal

  def _advance_targets(self) -> None:
    for enemy in self.enemies:
      if self.throttle >= 3 or self.steps % 3 == 0:
        enemy.column -= 1
      enemy.fire_in -= 1
      if enemy.fire_in == 0 and enemy.column >= 0:
        self.hostile.append(Target(enemy.row, enemy.column))
    for missile in self.hostile:
      missile.column -= 1
      if self.steps % 2 == 0:
        missile.row += int(np.sign(PLAYER_ROW - missile.row))

  def _advance_salvo(self) -> None:
    self.blasts = [(row, column, timer - 1)
                   for row, column, timer in self.blasts if timer > 1]
    remaining = []
    for shot in self.shots:
      shot.age += 1
      shot.column += 1 if shot.age == 1 else int(np.sign(shot.target.column - shot.column))
      if shot.age > 1:
        shot.row += int(np.sign(shot.target.row - shot.row))
      distance = max(abs(shot.row - shot.target.row), abs(shot.column - shot.target.column))
      if distance <= 1 or shot.age >= 12:
        self.blasts.append((shot.row, max(0, min(9, shot.column)), 3))
      else:
        remaining.append(shot)
    self.shots = remaining

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[PLAYER_ROW, self.column, 0] = True
    for channel, targets in ((1, self.hostile), (2, self.enemies), (3, self.shots)):
      for target in targets:
        if 0 <= target.column < 10:
          state[target.row, target.column, channel] = True
    for row, column, _ in self.blasts:
      state[max(0, row - 1):row + 2, max(0, column - 1):column + 2, 4] = True
    values = (self.steps, self.throttle, self.spawn_in, self.multiplier,
              int(self.last_action == 5), int(not (self.shots or self.blasts)))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
