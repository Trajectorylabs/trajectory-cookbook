from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4


@dataclass
class Board:
  x: float
  y: float
  slope: int


@dataclass
class Passenger:
  x: float
  y: float
  kind: int
  vx: float = 0
  vy: float = 0
  bounces: int = 0


class BoardingEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.cooldown = 0
    self.pressed = self.ready = self.terminal = False
    self.x, self.direction, self.kind = 50.0, -1, -1
    self.gate = self.gate_target = 91.0
    self.board_credit = 0
    self.boards: list[Board] = []
    self.passengers: list[Passenger] = []
    for y in range(47, 88, 2):
      self._add_board(y)

  def _add_board(self, y: float) -> None:
    x = self.random.uniform(9, 90)
    if all(math.hypot(board.x - x, board.y - y) >= 16 for board in self.boards):
      self.boards.append(Board(x, y, int(self.random.choice((-1, 1)))))

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    pressed = self.pressed if self.random.random() < self.sticky_action_prob else action == 5
    released = self.pressed and not pressed
    self.pressed = pressed
    self.steps += 1
    reward = 0
    for frame in range(FRAMES_PER_STEP):
      reward += self._advance_frame(released and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, released: bool) -> int:
    for board in self.boards:
      board.y -= 0.1
    self.boards = [board for board in self.boards if 15 < board.y < self.gate]
    self.gate_target = max(11, self.gate_target - 0.1)
    self.gate += (self.gate_target - self.gate) * 0.1
    if self.gate <= 12:
      self.terminal = True
      return 0
    if self.board_credit >= 2:
      self._add_board(self.gate - 3)
      self.board_credit -= 2

    self.cooldown = max(0, self.cooldown - 1)
    boarding = self.pressed and self.ready
    if boarding and not self.cooldown:
      self.passengers.append(Passenger(self.x, 9, self.kind))
      self.cooldown = 10
    self.x += self.direction * (0.1 if boarding else 1)
    if self.x < 10 or self.x > 90:
      self.direction = 1 if self.x < 10 else -1
    if released:
      if self.ready:
        self.kind *= -1
      self.ready = True

    reward = 0
    survivors = []
    for passenger in self.passengers:
      self._move_passenger(passenger)
      if passenger.y <= self.gate:
        survivors.append(passenger)
      elif (passenger.x - 50) * passenger.kind > 0:
        reward += passenger.bounces
        self.gate_target = min(91, self.gate_target + 2 * passenger.bounces)
        self.board_credit += passenger.bounces
      else:
        reward -= 1 + passenger.bounces
        self.gate_target -= min(20, math.sqrt(1 + passenger.bounces))
    self.passengers = survivors
    return reward

  def _move_passenger(self, passenger: Passenger) -> None:
    old_x, old_y = passenger.x, passenger.y
    passenger.vx *= 0.9
    passenger.vy = (passenger.vy + 0.2) * 0.9
    passenger.x += passenger.vx
    passenger.y += passenger.vy
    for board in self.boards:
      previous = old_y - board.y - board.slope * (old_x - board.x)
      current = passenger.y - board.y - board.slope * (passenger.x - board.x)
      if previous > 0 or current < 0 or current == previous:
        continue
      crossing_x = old_x + (passenger.x - old_x) * (-previous / (current - previous))
      if abs(crossing_x - board.x) <= 4:
        impulse = math.hypot(passenger.vx, passenger.vy) * math.sqrt(2)
        passenger.vx += board.slope * impulse
        passenger.vy += impulse
        passenger.x += 2 * passenger.vx
        passenger.y += 2 * passenger.vy
        passenger.bounces += 1
        break
    if passenger.x < 0 or passenger.x > 99:
      passenger.x = min(99, max(0, passenger.x))
      passenger.vx *= -1

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[0, int(self.x) // 10, 0] = True
    state[int(self.gate) // 10, :, 1] = True
    for board in self.boards:
      state[int(board.y) // 10, int(board.x) // 10, 2 if board.slope < 0 else 3] = True
    for passenger in self.passengers:
      state[int(passenger.y) // 10, int(passenger.x) // 10, 4 if passenger.kind < 0 else 5] = True
    state[0, :4, 1] = self.kind > 0, self.direction > 0, self.ready, self.pressed
    return state
