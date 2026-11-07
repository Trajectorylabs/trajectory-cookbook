from dataclasses import dataclass

import numpy as np


HORIZON = 400
WAVE_DELAY = 8
MAX_WAVE = 4
PERIMETER = (
  tuple((9, x) for x in range(10))
  + tuple((y, 9) for y in range(8, -1, -1))
  + tuple((0, x) for x in range(8, -1, -1))
  + tuple((y, 0) for y in range(1, 9))
)


@dataclass
class Ball:
  row: int
  column: int
  dy: int
  dx: int
  size: int = 3
  immunity: int = 0


class BCannonEnvironment:
  score_cap = HORIZON // WAVE_DELAY * MAX_WAVE * 8 * 99

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.held = self.wave_size = self.ball_index = 0
    self.perimeter_index, self.direction = 5, 1
    self.wave_max, self.ball_period, self.wave_wait = 2, 3, WAVE_DELAY
    self.balls: list[Ball] = []
    self.rope: list[tuple[int, int]] = []
    self.rope_length = 0
    self.terminal = False

  @property
  def player(self) -> tuple[int, int]:
    return PERIMETER[self.perimeter_index]

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _aim(self) -> tuple[int, int] | None:
    if not self.balls:
      return None
    row, column = self.player
    ball = min(self.balls, key=lambda item: (item.row - row) ** 2 + (item.column - column) ** 2)
    return max(1, min(8, ball.row + ball.dy)), max(1, min(8, ball.column + ball.dx))

  def _fire(self, target: tuple[int, int]) -> None:
    row, column = self.player
    dy, dx = target[0] - row, target[1] - column
    scale = max(abs(dy), abs(dx))
    self.rope = []
    for distance in range(1, 11):
      cell = row + round(dy * distance / scale), column + round(dx * distance / scale)
      if not all(0 < coordinate < 9 for coordinate in cell):
        break
      self.rope.append(cell)
    self.rope_length = 1

  def _spawn_wave(self) -> None:
    self.wave_size += 1
    if self.wave_size > self.wave_max:
      self.wave_size = 1
      self.wave_max = min(MAX_WAVE, self.wave_max + 1)
      self.ball_period = max(1, self.ball_period - 1)
    start = int(self.random.integers(1, 3))
    self.balls = [Ball(3, start + 2 * index, 1, 0) for index in range(self.wave_size)]
    self.ball_index = (self.ball_index + self.wave_size) % 100
    self.wave_wait = WAVE_DELAY

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    if action == 5 and self.last_action != 5:
      self.direction *= -1
    self.last_action = action
    self.steps += 1
    if action == 5:
      if self.balls:
        self.held += 1
        target = self._aim()
        if self.held == 3 and target is not None:
          self._fire(target)
    else:
      self.held = 0
      self.perimeter_index = (self.perimeter_index + self.direction) % len(PERIMETER)
    reward = self._advance_balls()
    if self.rope:
      self.rope_length += 1
      if self.rope_length > len(self.rope) + 1:
        self.rope.clear()
        self.rope_length = 0
    if not self.balls:
      self.wave_wait -= 1
      if self.wave_wait == 0:
        self._spawn_wave()
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_balls(self) -> int:
    reward = 0
    remaining = []
    rope = set(self.rope[:self.rope_length])
    for ball in self.balls:
      if self.steps % self.ball_period == 0:
        if not 0 <= ball.row + ball.dy < 10:
          ball.dy *= -1
        if not 0 <= ball.column + ball.dx < 10:
          ball.dx *= -1
        ball.row += ball.dy
        ball.column += ball.dx
      ball.immunity = max(0, ball.immunity - 1)
      cell = ball.row, ball.column
      if cell in rope and not ball.immunity:
        if ball.size == 0:
          reward += self.ball_index
        else:
          remaining.extend(Ball(*cell, -ball.dy or 1, dx, ball.size - 1, 3) for dx in (-1, 1))
          self.ball_index = (self.ball_index + 2) % 100
      else:
        self.terminal |= cell == self.player
        remaining.append(ball)
    self.balls = remaining
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for ball in self.balls:
      state[ball.row, ball.column, 1] = True
      state[ball.row, ball.column, 2] |= ball.immunity > 0
    for cell in self.rope[:self.rope_length]:
      state[*cell, 3] = True
    target = self._aim()
    if target is not None:
      state[*target, 4] = True
    values = (self.steps, self.held, self.wave_size, self.wave_max, self.wave_wait,
              self.ball_period, self.ball_index, int(self.direction > 0), int(self.last_action == 5))
    for row, value in enumerate(values):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
