import math

import numpy as np


WEATHER = ("clear", "fog", "rain", "snow")
SYSTEMS = ("video", "cannon", "radar", "treads")


class RobotTankEnvironment:
  """Twelve first-person squadrons with radar, weather, and system damage."""

  score_cap = 144.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 5_000):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = horizon
    self.random = np.random.RandomState()
    self.seed(0)
    self.reset()

  @classmethod
  def normalize_score(cls, raw_score: float) -> float:
    return min(max(raw_score, 0.0), cls.score_cap) / cls.score_cap

  def seed(self, seed: int) -> None:
    self.random.seed(seed)

  def reset(self) -> None:
    self.score = 0.0
    self.kills = 0
    self.reserves = 3
    self.steps = 0
    self.last_action = 0
    self.heading = 0
    self.previous_heading = 0
    self.damage: set[str] = set()
    self.rocket_distance: int | None = None
    self.shot_timer = 0
    self.terminal = False
    self._spawn_enemy()

  @property
  def squadron(self) -> int:
    return self.kills // 12

  @property
  def weather(self) -> str:
    return WEATHER[(self.squadron + self.steps // 240) % len(WEATHER)]

  @property
  def relative_bearing(self) -> int:
    return (self.enemy_bearing - self.heading + 8) % 16 - 8

  def _spawn_enemy(self) -> None:
    self.enemy_bearing = int(self.random.randint(16))
    self.previous_enemy_bearing = self.enemy_bearing
    self.enemy_distance = 5 + int(self.random.randint(4))
    self.previous_enemy_distance = self.enemy_distance
    self.rocket_distance = None

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_heading = self.heading
    self.previous_enemy_bearing = self.enemy_bearing
    self.previous_enemy_distance = self.enemy_distance
    movement_allowed = "treads" not in self.damage or self.steps % 3 == 0
    weather_allows_movement = self.weather != "rain" or self.steps % 2 == 0
    if action == 1 and movement_allowed:
      self.heading = (self.heading - 1) % 16
    elif action == 3 and movement_allowed:
      self.heading = (self.heading + 1) % 16
    elif action == 2 and movement_allowed and weather_allows_movement:
      self.enemy_distance = max(1, self.enemy_distance - 1)
    elif action == 4 and movement_allowed and weather_allows_movement:
      self.enemy_distance = min(9, self.enemy_distance + 1)
    reward = self._fire() if action == 5 else 0.0
    if reward == 0.0:
      self._enemy_turn()
      self._advance_rocket()
    self.score += reward
    self.steps += 1
    self.shot_timer = max(0, self.shot_timer - 1)
    if self.kills >= 144 or self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _fire(self) -> float:
    if "cannon" in self.damage and self.random.rand() < 0.5:
      return 0.0
    self.shot_timer = 2
    if self.rocket_distance is not None and abs(self.relative_bearing) <= 1:
      self.rocket_distance = None
      return 0.0
    if abs(self.relative_bearing) > 1:
      return 0.0
    self.kills += 1
    if self.kills % 12 == 0:
      self.reserves = min(12, self.reserves + 1)
    if self.kills < 144:
      self._spawn_enemy()
    return 1.0

  def _enemy_turn(self) -> None:
    if self.steps % max(2, 5 - self.squadron // 4) == 0:
      turn = int(np.sign((self.heading - self.enemy_bearing + 8) % 16 - 8))
      self.enemy_bearing = (self.enemy_bearing + turn) % 16
    if self.steps % 4 == 0:
      self.enemy_distance = max(1, self.enemy_distance - 1)
    if (
      self.rocket_distance is None
      and abs(self.relative_bearing) <= 1
      and self.enemy_distance <= 6
      and self.steps % max(5, 11 - self.squadron // 2) == 0
    ):
      self.rocket_distance = self.enemy_distance

  def _advance_rocket(self) -> None:
    if self.rocket_distance is None:
      return
    self.rocket_distance -= 1
    if self.rocket_distance > 0:
      return
    self.rocket_distance = None
    self._take_hit()

  def _take_hit(self) -> None:
    undamaged = [system for system in SYSTEMS if system not in self.damage]
    if undamaged:
      self.damage.add(undamaged[int(self.random.randint(len(undamaged)))])
      return
    if self.reserves == 0:
      self.terminal = True
      return
    self.reserves -= 1
    self.damage.clear()
    self.heading = 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[5, 5, 0] = True
    angle = self.relative_bearing * math.pi / 8
    radius = min(4, max(1, self.enemy_distance // 2))
    radar_row = int(round(5 - math.cos(angle) * radius))
    radar_column = int(round(5 + math.sin(angle) * radius))
    state[radar_row, radar_column, 1] = True
    previous_relative = (self.previous_enemy_bearing - self.previous_heading + 8) % 16 - 8
    previous_angle = previous_relative * math.pi / 8
    previous_radius = min(4, max(1, self.previous_enemy_distance // 2))
    previous_row = int(round(5 - math.cos(previous_angle) * previous_radius))
    previous_column = int(round(5 + math.sin(previous_angle) * previous_radius))
    state[previous_row, previous_column, 4] = True
    hour = (self.steps // 10) % 24
    visible = 5 <= hour < 20 and self.weather != "fog" and not (
      "video" in self.damage and self.steps % 4 < 2
    )
    if visible and abs(self.relative_bearing) <= 2:
      width = max(1, 4 - self.enemy_distance // 3)
      state[2:5, 5 - width : 6 + width, 1] = True
    if self.rocket_distance is not None:
      state[min(9, 4 + self.rocket_distance), 5, 1] = True
    if self.shot_timer > 0:
      state[4, 5, 3] = True
    for index, system in enumerate(SYSTEMS):
      state[0, index, 5] = system in self.damage
    values = (
      (self.heading, 4),
      (self.enemy_bearing, 4),
      (self.enemy_distance, 4),
      (self.kills, 8),
      (self.reserves, 4),
      (self.steps % 240, 8),
      (WEATHER.index(self.weather), 2),
    )
    offset = 4
    for value, width in values:
      for bit in range(width):
        index = offset + bit
        state[index // 10, index % 10, 5] = bool(value & (1 << bit))
      offset += width
    if self.rocket_distance is not None:
      for bit in range(4):
        index = offset + bit
        state[index // 10, index % 10, 5] = bool(self.rocket_distance & (1 << bit))
    return state
