from dataclasses import dataclass

import numpy as np


HORIZON = 400
GROUND = 32
UNIT = 4
SPAWN_INTERVAL = 5
MAX_BALLS = (HORIZON + SPAWN_INTERVAL - 1) // SPAWN_INTERVAL


@dataclass
class Ball:
  x: int
  y: int
  vx: int
  vy: int = 0


class BallsBombsEnvironment:
  score_cap = MAX_BALLS * (MAX_BALLS + 1) // 2

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.scroll = 0
    self.player = 2
    self.next_bomb = 6
    self.bomb: int | None = None
    self.armed = False
    self.blast: tuple[int, int] | None = None
    self.multiplier = 1
    self.balls: list[Ball] = []
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
    if action == 5:
      self.player += 1
    self.scroll = max(0, self.player - 4)
    if not self.armed and self.bomb is None and self.player >= self.next_bomb:
      self.bomb = self.scroll + 9
    if self.bomb == self.player:
      self.armed = True
      self.bomb = None
      self.next_bomb = self.player + int(self.random.integers(9, 13))
    if self.blast is not None:
      center, radius = self.blast
      self.blast = (center, radius + 1) if radius < 6 else None
      if self.blast is None:
        self.multiplier = 1
    if self.steps % SPAWN_INTERVAL == 1:
      from_left = self.random.random() < 0.1
      self.balls.append(Ball(self.scroll + (0 if from_left else 9),
                             int(self.random.integers(5, 17)), 1 if from_left else -1))
    reward = 0
    survivors = []
    for ball in self.balls:
      ball.vy += 1
      ball.y += ball.vy
      if ball.y >= GROUND:
        ball.y = GROUND
        ball.vy = -ball.vy
      if self.steps % 2 == 0:
        ball.x += ball.vx
      if ball.x <= self.scroll and ball.vx < 0:
        ball.vx = 2
      elif ball.x >= self.scroll + 9 and ball.vx > 0:
        ball.vx = -1
      if self._in_blast(ball.y // UNIT, ball.x):
        reward += self.multiplier
        self.multiplier += 1
      elif ball.x == self.player and ball.y // UNIT == GROUND // UNIT:
        if self.armed:
          self.armed = False
          self.blast = self.player, 1
        else:
          self.terminal = True
      elif ball.x >= self.scroll - 2:
        survivors.append(ball)
    self.balls = survivors
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _in_blast(self, row: int, x: int) -> bool:
    if self.blast is None:
      return False
    center, radius = self.blast
    distance_squared = (x - center) ** 2 + (GROUND // UNIT - row) ** 2
    return (radius - 0.75) ** 2 <= distance_squared <= (radius + 0.75) ** 2

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[GROUND // UNIT, self.player - self.scroll, 0] = True
    for ball in self.balls:
      column = ball.x - self.scroll
      if 0 <= column < 10:
        row = max(0, ball.y // UNIT)
        state[row, column, 1] = True
        state[row, column, 4] |= ball.vy < 0
    if self.bomb is not None and 0 <= self.bomb - self.scroll < 10:
      state[GROUND // UNIT, self.bomb - self.scroll, 2] = True
    if self.blast is not None:
      for row in range(9):
        for column in range(10):
          state[row, column, 3] = self._in_blast(row, column + self.scroll)
    values = (self.steps, self.multiplier, int(self.armed), self.player,
              self.next_bomb, self.last_action, 0 if self.blast is None else self.blast[1])
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
