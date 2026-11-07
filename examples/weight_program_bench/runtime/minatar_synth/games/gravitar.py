import numpy as np


HEADINGS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PLANET_LAYOUTS = (
  (7, 7, 6, 6, 7, 8, 8, 7, 7, 6),
  (8, 7, 7, 6, 5, 5, 6, 7, 8, 8),
  (6, 6, 7, 8, 8, 7, 6, 5, 5, 6),
)


class GravitarEnvironment:
  """Finite twelve-system gravity, fuel, bunker, and reactor campaign."""

  score_cap = 147_000.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 10_800):
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
    self.mode = "map"
    self.system = 0
    self.completed_planets: set[int] = set()
    self.selected_destination = 0
    self.lives = 6
    self.fuel = 10_000
    self.score = 0.0
    self.next_bonus_score = 10_000.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.player = [5, 5]
    self.previous_player = self.player.copy()
    self.heading = 0
    self.velocity = [0, 0]
    self.projectile: list[int] | None = None
    self.projectile_heading = 0
    self.shield_active = False
    self.enemy_projectiles: list[list[int]] = []
    self.bunkers: set[tuple[int, int]] = set()
    self.fuel_depots: set[tuple[int, int]] = set()
    self.terrain = PLANET_LAYOUTS[0]
    self.mission_steps = 0
    self.reactor_activated = False
    self.reactor_countdown = 0
    self.destroyed_bunkers: set[tuple[int, int, int, int]] = set()
    self.collected_depots: set[tuple[int, int, int, int]] = set()

  @property
  def reverse_gravity(self) -> bool:
    return (self.system // 3) % 2 == 1

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    reward = self._act_map(action) if self.mode == "map" else self._act_flight(action)
    self.score += reward
    while self.score >= self.next_bonus_score:
      self.lives = min(30, self.lives + 1)
      self.next_bonus_score += 10_000.0
    self.steps += 1
    if self.steps >= self.horizon or self.fuel <= 0 or self.lives <= 0:
      self.terminal = True
    return reward, self.terminal

  def _act_map(self, action: int) -> float:
    if action == 1:
      self.selected_destination = (self.selected_destination - 1) % 4
    elif action == 3:
      self.selected_destination = (self.selected_destination + 1) % 4
    elif action in (2, 4, 5):
      if self.selected_destination == 3:
        self._enter_reactor()
      elif self.selected_destination not in self.completed_planets:
        self._enter_planet(self.selected_destination)
    return 0.0

  def _enter_planet(self, planet: int) -> None:
    self.mode = "planet"
    self.terrain = PLANET_LAYOUTS[planet]
    bunker_columns = ((1, 5, 8), (0, 4, 9), (2, 6, 8))[planet]
    bunkers = {(self.terrain[column] - 1, column) for column in bunker_columns}
    self.bunkers = {
      bunker for bunker in bunkers if (self.system, planet, *bunker) not in self.destroyed_bunkers
    }
    depot_columns = ((3, 7), (2, 7), (1, 6))[planet]
    depots = {(self.terrain[column] - 1, column) for column in depot_columns}
    self.fuel_depots = {
      depot for depot in depots if (self.system, planet, *depot) not in self.collected_depots
    }
    self._reset_ship()

  def _enter_reactor(self) -> None:
    self.mode = "reactor"
    self.terrain = (9, 8, 7, 6, 5, 5, 6, 7, 8, 9)
    bunkers = {(6, 2), (6, 7)}
    self.bunkers = {
      bunker for bunker in bunkers if (self.system, 3, *bunker) not in self.destroyed_bunkers
    }
    self.fuel_depots = set()
    self.reactor_activated = False
    self.reactor_countdown = max(20, 60 - 3 * self.system)
    self._reset_ship()

  def _reset_ship(self) -> None:
    self.player = [1, 5]
    self.previous_player = self.player.copy()
    self.heading = 2
    self.velocity = [0, 0]
    self.projectile = None
    self.shield_active = False
    self.enemy_projectiles = []
    self.mission_steps = 0

  def _act_flight(self, action: int) -> float:
    self.previous_player = self.player.copy()
    self.shield_active = False
    if action == 1:
      self.heading = (self.heading - 1) % 4
    elif action == 3:
      self.heading = (self.heading + 1) % 4
    elif action == 2:
      self._thrust()
    elif action == 4:
      self.shield_active = self.fuel >= 10
      self._tractor()
    elif action == 5 and self.projectile is None:
      self.projectile = self.player.copy()
      self.projectile_heading = self.heading
    reward = self._move_projectiles()
    if self.mode == "map":
      return reward
    if self.mode == "reactor" and self.reactor_activated and self.player[0] == 0:
      return reward + self._advance_system()
    self._apply_gravity_and_move()
    if self.mode == "map":
      return reward
    reward += self._move_enemy_projectiles()
    if self.mode == "map":
      return reward
    self._fire_bunkers()
    self.mission_steps += 1
    if self.mode == "reactor" and self.reactor_activated:
      self.reactor_countdown -= 1
      if self.reactor_countdown <= 0:
        self._lose_ship()
    return reward

  def _thrust(self) -> None:
    if self.fuel < 25:
      return
    direction = HEADINGS[self.heading]
    self.velocity[0] = int(np.clip(self.velocity[0] + direction[0], -2, 2))
    self.velocity[1] = int(np.clip(self.velocity[1] + direction[1], -2, 2))
    self.fuel -= 25

  def _tractor(self) -> None:
    if self.fuel < 10:
      return
    self.fuel -= 10
    nearby = [depot for depot in self.fuel_depots if abs(depot[0] - self.player[0]) + abs(depot[1] - self.player[1]) <= 2]
    if nearby:
      depot = min(nearby)
      self.fuel_depots.remove(depot)
      self.collected_depots.add((self.system, self.selected_destination, *depot))
      self.fuel = min(20_000, self.fuel + 5_000)

  def _move_projectiles(self) -> float:
    if self.projectile is None:
      return 0.0
    direction = HEADINGS[self.projectile_heading]
    self.projectile[0] += direction[0]
    self.projectile[1] += direction[1]
    position = tuple(self.projectile)
    if position in self.bunkers:
      self.bunkers.remove(position)
      destination = self.selected_destination if self.mode == "planet" else 3
      self.destroyed_bunkers.add((self.system, destination, *position))
      self.projectile = None
      reward = 250.0
      if self.mode == "planet" and not self.bunkers:
        reward += self._complete_planet()
      return reward
    if self.mode == "reactor" and position == (4, 5):
      self.reactor_activated = True
      self.projectile = None
      return 0.0
    if not (0 <= position[0] < 10 and 0 <= position[1] < 10):
      self.projectile = None
    elif position[0] >= self.terrain[position[1]]:
      self.projectile = None
    return 0.0

  def _apply_gravity_and_move(self) -> None:
    gravity = -1 if self.reverse_gravity else 1
    if self.mission_steps % 2 == 0:
      self.velocity[0] = int(np.clip(self.velocity[0] + gravity, -2, 2))
    self.player[0] += int(np.sign(self.velocity[0]))
    self.player[1] += int(np.sign(self.velocity[1]))
    if self.player[0] < 0:
      if self.mode == "planet":
        self.player[0] = 0
        self.velocity[0] = 0
      else:
        self.player[0] = 0
      return
    if self.player[1] < 0 or self.player[1] > 9 or self.player[0] > 9:
      self._lose_ship()
      return
    if self.player[0] >= self.terrain[self.player[1]]:
      self._lose_ship()

  def _move_enemy_projectiles(self) -> float:
    moved = []
    for projectile in self.enemy_projectiles:
      projectile[0] += int(np.sign(self.player[0] - projectile[0]))
      projectile[1] += int(np.sign(self.player[1] - projectile[1]))
      if projectile == self.player:
        if self.shield_active:
          continue
        self._lose_ship()
        return 0.0
      if 0 <= projectile[0] < 10 and 0 <= projectile[1] < 10:
        moved.append(projectile)
    self.enemy_projectiles = moved
    return 0.0

  def _fire_bunkers(self) -> None:
    if self.mission_steps % 12 or not self.bunkers:
      return
    bunker = min(self.bunkers, key=lambda item: abs(item[1] - self.player[1]))
    self.enemy_projectiles.append(list(bunker))

  def _complete_planet(self) -> float:
    planet = self.selected_destination
    self.completed_planets.add(planet)
    bonus = float(max(0, 2_000 - 10 * self.mission_steps))
    self.mode = "map"
    self.player = [5, 5]
    if len(self.completed_planets) == 3:
      bonus += self._advance_system()
    return bonus

  def _advance_system(self) -> float:
    self.system += 1
    self.completed_planets = set()
    self.mode = "map"
    self.player = [5, 5]
    self.selected_destination = 0
    self.fuel = min(20_000, self.fuel + 7_000)
    self.lives = min(30, self.lives + 2)
    if self.system >= 12:
      self.terminal = True
    return 4_000.0

  def _lose_ship(self) -> None:
    self.lives -= 1
    self.mode = "map"
    self.player = [5, 5]
    self.velocity = [0, 0]
    self.projectile = None
    self.enemy_projectiles = []
    if self.lives <= 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for bunker in self.bunkers:
      state[*bunker, 1] = True
    for projectile in self.enemy_projectiles:
      state[*projectile, 1] = True
    for depot in self.fuel_depots:
      state[*depot, 2] = True
    if self.mode == "reactor" and not self.reactor_activated:
      state[4, 5, 2] = True
    if self.projectile is not None:
      state[*self.projectile, 3] = True
    if self.shield_active:
      state[*self.player, 3] = True
    state[*self.previous_player, 4] = True
    if self.mode != "map":
      for column, floor in enumerate(self.terrain):
        state[floor:, column, 4] = True
    for bit in range(4):
      state[9, bit, 5] = bool(self.system & (1 << bit))
    for bit in range(5):
      state[8, bit, 5] = bool(self.lives & (1 << bit))
    fuel_bucket = min(31, self.fuel // 625)
    for bit in range(5):
      state[7, bit, 5] = bool(fuel_bucket & (1 << bit))
    state[6, self.heading, 5] = True
    state[5, self.selected_destination, 5] = True
    for planet in self.completed_planets:
      state[4, planet, 5] = True
    state[3, self.velocity[0] + 2, 5] = True
    state[2, self.velocity[1] + 2, 5] = True
    state[1, 0, 5] = self.mode == "map"
    state[1, 1, 5] = self.mode == "planet"
    state[1, 2, 5] = self.mode == "reactor"
    state[1, 3, 5] = self.reverse_gravity
    state[1, 4, 5] = self.reactor_activated
    return state
