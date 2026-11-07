import numpy as np


DIRECTIONS = (
  (-1, 0),
  (-1, 1),
  (0, 1),
  (1, 1),
  (1, 0),
  (1, -1),
  (0, -1),
  (-1, -1),
)
POINTS = {3: 20.0, 2: 50.0, 1: 100.0}


class AsteroidsEnvironment:
  """Momentum, wraparound, fragmentation, and risky hyperspace on a torus."""

  score_cap = 6_240.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_500):
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
    self.ship = [5, 5]
    self.previous_ship = self.ship.copy()
    self.velocity = [0, 0]
    self.direction = 0
    self.shots: list[list[int]] = []
    self.wave = 0
    self.lives = 4
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_wave()

  def _start_wave(self) -> None:
    starts = ((0, 0, 1, 1), (0, 9, 1, -1), (9, 0, -1, 1), (9, 9, -1, -1))
    self.asteroids = [
      {"row": row, "column": column, "dr": dr, "dc": dc, "size": 3}
      for row, column, dr, dc in starts
    ]
    self.previous_asteroids = [[int(asteroid["row"]), int(asteroid["column"])] for asteroid in self.asteroids]

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_ship = self.ship.copy()
    if action == 1:
      self.direction = (self.direction - 1) % 8
    elif action == 3:
      self.direction = (self.direction + 1) % 8
    elif action == 2:
      thrust = DIRECTIONS[self.direction]
      self.velocity[0] = int(np.clip(self.velocity[0] + thrust[0], -1, 1))
      self.velocity[1] = int(np.clip(self.velocity[1] + thrust[1], -1, 1))
    elif action == 4:
      self.ship = [int(self.random.randint(10)), int(self.random.randint(10))]
      self.velocity = [0, 0]
    elif action == 5:
      dr, dc = DIRECTIONS[self.direction]
      self.shots.append([self.ship[0], self.ship[1], dr, dc, 12])

    if action != 4:
      self.ship[0] = (self.ship[0] + self.velocity[0]) % 10
      self.ship[1] = (self.ship[1] + self.velocity[1]) % 10
    self.previous_asteroids = [[int(asteroid["row"]), int(asteroid["column"])] for asteroid in self.asteroids]
    self._move_asteroids()
    self._move_shots()
    reward = self._resolve_shots()
    if any([asteroid["row"], asteroid["column"]] == self.ship for asteroid in self.asteroids):
      self._lose_ship()
    self.steps += 1
    if not self.asteroids:
      self.wave += 1
      if self.wave == 3:
        self.terminal = True
      else:
        self._start_wave()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_asteroids(self) -> None:
    for asteroid in self.asteroids:
      asteroid["row"] = (int(asteroid["row"]) + int(asteroid["dr"])) % 10
      asteroid["column"] = (int(asteroid["column"]) + int(asteroid["dc"])) % 10

  def _move_shots(self) -> None:
    for shot in self.shots:
      shot[0] = (shot[0] + shot[2]) % 10
      shot[1] = (shot[1] + shot[3]) % 10
      shot[4] -= 1
    self.shots = [shot for shot in self.shots if shot[4] > 0]

  def _resolve_shots(self) -> float:
    reward = 0.0
    remaining_shots = []
    for shot in self.shots:
      hit = next(
        (asteroid for asteroid in self.asteroids if [asteroid["row"], asteroid["column"]] == shot[:2]),
        None,
      )
      if hit is None:
        remaining_shots.append(shot)
        continue
      self.asteroids.remove(hit)
      size = int(hit["size"])
      reward += POINTS[size]
      if size > 1:
        for turn in (-1, 1):
          direction = DIRECTIONS.index((int(hit["dr"]), int(hit["dc"])))
          dr, dc = DIRECTIONS[(direction + turn) % 8]
          self.asteroids.append({
            "row": int(hit["row"]),
            "column": int(hit["column"]),
            "dr": dr,
            "dc": dc,
            "size": size - 1,
          })
    self.shots = remaining_shots
    return reward

  def _lose_ship(self) -> None:
    self.lives -= 1
    self.ship = [5, 5]
    self.previous_ship = self.ship.copy()
    self.velocity = [0, 0]
    self.shots.clear()
    if self.lives <= 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.ship, 0] = True
    for asteroid in self.asteroids:
      row, column = int(asteroid["row"]), int(asteroid["column"])
      state[row, column, 1] = True
      state[row, column, 2] = True
      size = int(asteroid["size"])
      state[row, (column + size - 1) % 10, 5] = True
    for shot in self.shots:
      state[shot[0], shot[1], 3] = True
    for row, column in self.previous_asteroids:
      state[row, column, 4] = True
    state[*self.previous_ship, 4] = True
    state[9, : self.lives, 5] = True
    state[8, : self.wave + 1, 5] = True
    velocity_marker = (self.velocity[0] + 1) * 3 + self.velocity[1] + 1
    state[7, velocity_marker, 5] = True
    state[6, self.direction, 5] = True
    return state
