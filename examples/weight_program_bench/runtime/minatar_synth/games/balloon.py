from dataclasses import dataclass
from math import hypot

import numpy as np


HORIZON = 400
BONUS_INTERVAL = 8
BALLOON_INTERVAL = 28
MAX_BONUSES = (HORIZON + BONUS_INTERVAL - 1) // BONUS_INTERVAL
MOVES = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


@dataclass
class Balloon:
  row: float
  column: float
  drift: float
  rise: float = -0.24
  radius: float = 1.8


@dataclass
class Bonus:
  row: float
  column: float
  balloon: Balloon | None = None


class BalloonEnvironment:
  score_cap = MAX_BONUSES * (MAX_BONUSES + 1) // 2

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.cursor = (5, 5)
    self.wind = (0.0, 0.0)
    self.balloons: list[Balloon] = []
    self.bonuses: list[Bonus] = []
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    row, column = self.cursor
    dr, dc = MOVES.get(action, (0, 0))
    self.cursor = max(0, min(9, row + dr)), max(0, min(9, column + dc))
    self.wind = (0.5 * self.wind[0] + 0.25 * (self.cursor[0] - row),
                 0.5 * self.wind[1] + 0.25 * (self.cursor[1] - column))
    if self.steps % BALLOON_INTERVAL == 1:
      self.balloons.append(Balloon(9.5, float(self.random.uniform(2, 7)),
                                   float(self.random.uniform(-0.12, 0.12))))
    if self.steps % BONUS_INTERVAL == 1:
      self.bonuses.append(Bonus(0.0, float(self.random.uniform(1, 8))))
    for balloon in self.balloons:
      if hypot(balloon.row - self.cursor[0], balloon.column - self.cursor[1]) < balloon.radius:
        balloon.rise += self.wind[0]
        balloon.drift += self.wind[1]
      balloon.row += balloon.rise
      balloon.column += balloon.drift
      if not balloon.radius <= balloon.column <= 9 - balloon.radius:
        balloon.column = max(balloon.radius, min(9 - balloon.radius, balloon.column))
        balloon.drift *= -0.7
      if balloon.row > 10 - balloon.radius and balloon.rise > 0:
        balloon.rise *= -0.7
      balloon.drift *= 0.9
      balloon.rise += 0.15 * (-0.24 - balloon.rise)
      balloon.radius -= 0.015 * hypot(balloon.drift, balloon.rise)
    self.balloons = [balloon for balloon in self.balloons
                     if balloon.row >= balloon.radius and balloon.radius >= 0.8]
    alive = {id(balloon) for balloon in self.balloons}
    collected = 0
    remaining = []
    for bonus in self.bonuses:
      if bonus.balloon is None:
        bonus.row += 0.22
        for balloon in self.balloons:
          if hypot(balloon.row - bonus.row, balloon.column - bonus.column) < 1.3 * balloon.radius:
            bonus.balloon = balloon
            break
        if bonus.balloon is None and bonus.row >= 10:
          self.terminal = True
      if bonus.balloon is not None:
        if id(bonus.balloon) not in alive:
          collected += 1
          continue
        bonus.row, bonus.column = bonus.balloon.row, bonus.balloon.column
      remaining.append(bonus)
    self.bonuses = remaining
    self.terminal |= self.steps >= self.horizon
    return float(collected * (collected + 1) // 2), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.cursor, 0] = True
    rows, columns = np.ogrid[:10, :10]
    for balloon in self.balloons:
      state[:, :, 1] |= (rows - balloon.row) ** 2 + (columns - balloon.column) ** 2 <= balloon.radius ** 2
      state[max(0, min(9, round(balloon.row + balloon.rise))),
            max(0, min(9, round(balloon.column + balloon.drift))), 4] = True
    for bonus in self.bonuses:
      state[max(0, min(9, round(bonus.row))), round(bonus.column), 3 if bonus.balloon else 2] = True
    values = (self.steps, round(self.wind[0] * 64) + 32,
              round(self.wind[1] * 64) + 32, self.last_action)
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
