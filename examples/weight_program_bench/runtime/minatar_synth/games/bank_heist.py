import numpy as np


ROADS = np.array([
  [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
  [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
  [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
  [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
  [1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
  [1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
  [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
  [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
], dtype=bool)
BANK_LOCATIONS = ((1, 2), (1, 6), (3, 4), (3, 8), (5, 2), (5, 6), (7, 4), (7, 8), (9, 2))
TOWN_BONUSES = (93.0, 186.0, 279.0, 372.0)
DELTAS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


class BankHeistEnvironment:
  """Four-town robbery route with fuel, pursuing cops, and tailpipe dynamite."""

  score_cap = 4_530.0

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
    self.town = 0
    self.cars = 5
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self._start_town(full_fuel=True)

  def _start_town(self, full_fuel: bool) -> None:
    previous_fuel = getattr(self, "fuel", 100)
    self.fuel = 100 if full_fuel else max(previous_fuel, 50)
    self.player = [9, 0]
    self.previous_player = self.player.copy()
    self.direction = 3
    self.cops: list[dict[str, object]] = []
    self.previous_cops: list[list[int]] = []
    self.dynamite: list[dict[str, int]] = []
    self.robbed = 0
    self.spawned_banks = 3
    self.banks = set(BANK_LOCATIONS[:3])

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    if action in DELTAS:
      self.direction = action
      self._move_player(action)
    elif action == 5 and self.fuel > 5:
      row_delta, column_delta = DELTAS[self.direction]
      self.dynamite.append({
        "row": (self.player[0] - row_delta) % 10,
        "column": (self.player[1] - column_delta) % 10,
        "timer": 2,
      })
      self.fuel -= 5

    reward = self._rob_bank()
    self.previous_cops = [list(cop["position"]) for cop in self.cops]
    self._move_cops()
    reward += self._detonate_dynamite()
    if any(list(cop["position"]) == self.player for cop in self.cops):
      self._lose_car()
    self.fuel -= 1
    if self.fuel <= 0:
      self._lose_car()
    if self.player == [0, 9]:
      reward += self._leave_town()
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    row_delta, column_delta = DELTAS[action]
    target = [self.player[0] + row_delta, self.player[1] + column_delta]
    if self.player[0] in (3, 7) and target[1] < 0:
      target[1] = 9
    elif self.player[0] in (3, 7) and target[1] > 9:
      target[1] = 0
    if 0 <= target[0] < 10 and 0 <= target[1] < 10 and ROADS[*target]:
      self.player = target

  def _rob_bank(self) -> float:
    position = tuple(self.player)
    if position not in self.banks:
      return 0.0
    self.banks.remove(position)
    self.robbed += 1
    if len(self.cops) < 3:
      self.cops.append({"position": [*position], "previous": [*position]})
    return float(self.robbed * 10)

  def _move_cops(self) -> None:
    for cop in self.cops:
      position = list(cop["position"])
      previous = list(cop["previous"])
      candidates = []
      for action in DELTAS:
        row_delta, column_delta = DELTAS[action]
        target = [position[0] + row_delta, position[1] + column_delta]
        if position[0] in (3, 7) and target[1] < 0:
          target[1] = 9
        elif position[0] in (3, 7) and target[1] > 9:
          target[1] = 0
        if 0 <= target[0] < 10 and 0 <= target[1] < 10 and ROADS[*target] and target != previous:
          distance = abs(target[0] - self.player[0]) + abs(target[1] - self.player[1])
          candidates.append((distance, action, target))
      if candidates:
        cop["previous"] = position
        cop["position"] = min(candidates)[2]

  def _detonate_dynamite(self) -> float:
    reward = 0.0
    remaining = []
    for stick in self.dynamite:
      stick["timer"] -= 1
      if stick["timer"] > 0:
        remaining.append(stick)
        continue
      blast = [stick["row"], stick["column"]]
      if blast == self.player:
        self._lose_car()
      pursuit_count = len(self.cops)
      value = {1: 10.0, 2: 30.0, 3: 50.0}.get(pursuit_count, 0.0)
      destroyed = [cop for cop in self.cops if list(cop["position"]) == blast]
      for cop in destroyed:
        self.cops.remove(cop)
        reward += value
        if self.spawned_banks < 9:
          self.banks.add(BANK_LOCATIONS[self.spawned_banks])
          self.spawned_banks += 1
    self.dynamite = remaining
    return reward

  def _lose_car(self) -> None:
    self.cars -= 1
    self.player = [9, 0]
    self.previous_player = self.player.copy()
    self.fuel = 100
    self.cops.clear()
    self.dynamite.clear()
    if self.cars <= 0:
      self.terminal = True

  def _leave_town(self) -> float:
    reward = TOWN_BONUSES[self.town] if self.robbed >= 9 else 0.0
    full_fuel = self.robbed >= 9
    if full_fuel:
      self.cars += 1
    self.town += 1
    if self.town == 4:
      self.terminal = True
    else:
      self._start_town(full_fuel=full_fuel)
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for cop in self.cops:
      state[*list(cop["position"]), 1] = True
    for bank in self.banks:
      state[*bank, 2] = True
    state[0, 9, 2] = True
    for stick in self.dynamite:
      state[stick["row"], stick["column"], 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_cops:
      state[*position, 4] = True
    state[9, : min(10, (self.fuel + 9) // 10), 5] = True
    state[8, : min(10, self.cars), 5] = True
    state[7, : self.town + 1, 5] = True
    state[6, : min(9, self.robbed), 5] = True
    return state
