import math

import numpy as np


class EnduroEnvironment:
  """Five-day endurance race with traffic quotas and changing visibility."""

  score_cap = 1_750.0

  def __init__(self, sticky_action_prob: float = 0.1, day_length: int = 350):
    self.sticky_action_prob = sticky_action_prob
    self.day_length = day_length
    self.horizon = day_length * 5
    self.random = np.random.RandomState()
    self.seed(0)
    self.reset()

  @classmethod
  def normalize_score(cls, raw_score: float) -> float:
    return min(max(raw_score, 0.0), cls.score_cap) / cls.score_cap

  def seed(self, seed: int) -> None:
    self.random.seed(seed)

  def reset(self) -> None:
    self.player = [8, 5]
    self.previous_player = self.player.copy()
    self.speed = 0
    self.day = 1
    self.day_step = 0
    self.steps = 0
    self.passed = 0
    self.remaining_quota = 200
    self.traffic: list[dict[str, object]] = []
    self.previous_traffic: list[list[int]] = []
    self.last_action = 0
    self.terminal = False
    self.collision = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_traffic = [list(car["position"]) for car in self.traffic]
    self.collision = False
    self._apply_action(action)
    self._spawn_traffic()
    reward = self._move_traffic()
    self._resolve_collisions()
    self.day_step += 1
    self.steps += 1
    if self.day_step >= self.day_length:
      self._finish_day()
    return reward, self.terminal

  def _apply_action(self, action: int) -> None:
    if action == 5:
      self.speed = min(5, self.speed + 1)
    elif action == 4:
      self.speed = max(0, self.speed - 1)
    elif action in (1, 3):
      if self.weather == "ice" and self.day_step % 2:
        return
      self.player[1] += -1 if action == 1 else 1
      left, right = self.road_bounds(8)
      self.player[1] = int(np.clip(self.player[1], left, right))

  def _spawn_traffic(self) -> None:
    if self.speed == 0:
      return
    interval = max(1, 6 - self.speed)
    if self.day_step % interval:
      return
    left, right = self.road_bounds(0)
    occupied = {car["position"][1] for car in self.traffic if car["position"][0] <= 1}
    lanes = [column for column in range(left, right + 1) if column not in occupied]
    if not lanes:
      return
    self.traffic.append(
      {
        "position": [0, int(self.random.choice(lanes))],
        "speed": 1 + int(self.random.randint(2)),
      }
    )

  def _move_traffic(self) -> float:
    reward = 0.0
    survivors = []
    relative_speed = max(0, self.speed - 1)
    for car in self.traffic:
      movement = max(0, relative_speed - int(car["speed"]) + 1)
      car["position"][0] += movement
      row = min(9, car["position"][0])
      left, right = self.road_bounds(row)
      car["position"][1] = int(np.clip(car["position"][1], left, right))
      if car["position"][0] > 9:
        reward += 1.0
        self.passed += 1
        self.remaining_quota = max(0, self.remaining_quota - 1)
      else:
        survivors.append(car)
    self.traffic = survivors
    return reward

  def _resolve_collisions(self) -> None:
    collided = [car for car in self.traffic if car["position"] == self.player]
    if not collided:
      return
    self.collision = True
    self.speed = 1
    collided_ids = {id(car) for car in collided}
    self.traffic = [car for car in self.traffic if id(car) not in collided_ids]

  def _finish_day(self) -> None:
    if self.remaining_quota > 0 or self.day == 5:
      self.terminal = True
      return
    self.day += 1
    self.day_step = 0
    self.remaining_quota = 300
    self.traffic = []

  @property
  def weather(self) -> str:
    if self.day_step < 50:
      return "dawn"
    if self.day_step < 150:
      return "day"
    if self.day_step < 220:
      return "ice"
    if self.day_step < 280:
      return "fog"
    return "night"

  def road_bounds(self, row: int) -> tuple[int, int]:
    phase = (self.day_step + (9 - row) * 4) / 24.0
    center = 5 + int(round(2 * math.sin(phase)))
    return max(0, center - 3), min(9, center + 3)

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for row in range(10):
      left, right = self.road_bounds(row)
      state[row, left : right + 1, 2] = True
    for car in self.traffic:
      position = car["position"]
      visible = self.weather not in ("fog", "night") or position[0] >= 4
      if visible:
        state[*position, 1] = True
      if self.weather not in ("fog", "night"):
        state[*position, 2] = True
      else:
        state[position[0], position[1], 3] = visible
    state[*self.previous_player, 4] = True
    for position in self.previous_traffic:
      visible = self.weather not in ("fog", "night") or position[0] >= 4
      if visible and 0 <= position[0] < 10:
        state[*position, 4] = True
    state[9, : self.speed, 5] = True
    state[8, : self.day, 5] = True
    weather_index = ("dawn", "day", "ice", "fog", "night").index(self.weather)
    state[7, : weather_index + 1, 5] = True
    for bit in range(9):
      if self.remaining_quota & (1 << bit):
        state[6, bit, 5] = True
    if self.collision:
      state[5, :3, 5] = True
    return state
