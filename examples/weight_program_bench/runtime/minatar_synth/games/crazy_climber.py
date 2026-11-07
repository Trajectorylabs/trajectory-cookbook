import numpy as np


class CrazyClimberEnvironment:
  """Four-building two-phase climbing task with windows and falling hazards."""

  score_cap = 1_008_000.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 2_000):
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
    self.building = 1
    self.lives = 5
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.previous_player = [8, 4]
    self._start_building()

  def _start_building(self) -> None:
    self.player = [8, 4]
    self.previous_player = self.player.copy()
    self.reaching = False
    self.at_roof = False
    self.pickup_steps = 30
    self.helicopter_column = 0
    self.bonus = self.building * 100_000.0
    self.building_steps = 0
    self.electric_shocks = 0
    self.hazards: list[dict[str, object]] = []
    self.previous_hazards: list[list[int]] = []
    self._update_windows()

  def _update_windows(self) -> None:
    period = max(4, 9 - self.building)
    closed_span = 1 + self.building // 3
    self.windows = np.ones((8, 10), dtype=bool)
    for row in range(8):
      for column in range(10):
        phase = (self.building_steps + row * 3 + column * 5) % period
        self.windows[row, column] = phase >= closed_span

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_hazards = [list(hazard["position"]) for hazard in self.hazards]
    building_before_action = self.building
    reward = 0.0
    if self.at_roof:
      reward += self._act_on_roof(action)
    else:
      reward += self._act_on_wall(action)
    if self.building != building_before_action or self.terminal:
      self.steps += 1
      if self.steps >= self.horizon:
        self.terminal = True
      return reward, self.terminal
    self._move_hazards()
    self._update_windows()
    if not self.at_roof:
      self._resolve_window_grip(action)
      self._resolve_hazard_contact()
      self._resolve_electric_sign()
    self.building_steps += 1
    self.steps += 1
    if self.building_steps % 10 == 0:
      self.bonus = max(0.0, self.bonus - 100.0)
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _act_on_wall(self, action: int) -> float:
    if action in (1, 3) and not self.reaching:
      delta = -1 if action == 1 else 1
      self.player[1] = int(np.clip(self.player[1] + delta, 0, 8))
    elif action == 2:
      if self.reaching:
        self.player[0] -= 1
        self.reaching = False
        if self.player[0] == 0:
          self.at_roof = True
          self.pickup_steps = 30
        return self.building * 100.0
      self.reaching = True
    elif action == 4:
      self.reaching = False
    return 0.0

  def _act_on_roof(self, action: int) -> float:
    if action == 1:
      self.player[1] = max(0, self.player[1] - 1)
    elif action == 3:
      self.player[1] = min(9, self.player[1] + 1)
    self.helicopter_column = (self.helicopter_column + 1) % 10
    self.pickup_steps -= 1
    if self.pickup_steps <= 0:
      self.bonus = 0.0
    if action != 5 or abs(self.player[1] - self.helicopter_column) > 1:
      return 0.0
    reward = self.bonus
    self.building += 1
    if self.building > 4:
      self.terminal = True
    else:
      self._start_building()
    return reward

  def _move_hazards(self) -> None:
    spawn_period = max(5, 13 - 2 * self.building)
    if self.building_steps % spawn_period == 0:
      kinds = ["doctor", "egg"]
      if self.building >= 2:
        kinds.append("girder")
      kind = kinds[int(self.random.randint(len(kinds)))]
      self.hazards.append(
        {"position": [0, int(self.random.randint(10))], "kind": kind}
      )
    for hazard in self.hazards:
      hazard["position"][0] += 1
    self.hazards = [hazard for hazard in self.hazards if hazard["position"][0] <= 8]

  def _hand_positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
    row = max(0, self.player[0] - 1) if self.reaching else min(7, self.player[0])
    return (row, self.player[1]), (row, self.player[1] + 1)

  def _resolve_window_grip(self, action: int) -> None:
    hands = self._hand_positions()
    closed = sum(not self.windows[position] for position in hands)
    if closed == 2 or (closed == 1 and action in (1, 2, 3, 4)):
      self._fall()

  def _resolve_hazard_contact(self) -> None:
    contacts = [hazard for hazard in self.hazards if hazard["position"] == self.player]
    for hazard in contacts:
      self.bonus = max(0.0, self.bonus - 100.0)
      if self.reaching or hazard["kind"] == "girder":
        self._fall()
        break
    self.hazards = [hazard for hazard in self.hazards if hazard not in contacts]

  def _resolve_electric_sign(self) -> None:
    if self.building < 3 or self.player[0] != 3 or self.player[1] not in (7, 8):
      return
    self.electric_shocks += 1
    self.bonus = max(0.0, self.bonus - 100.0)
    if self.electric_shocks >= 11:
      self._fall()

  def _fall(self) -> None:
    self.lives -= 1
    self.reaching = False
    self.hazards = []
    self.electric_shocks = 0
    if self.lives <= 0:
      self.terminal = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for row in range(8):
      for column in range(10):
        if not self.windows[row, column]:
          state[row, column, 1] = True
        else:
          state[row, column, 2] = True
    for hazard in self.hazards:
      state[*hazard["position"], 1] = True
      if hazard["kind"] == "girder":
        state[*hazard["position"], 5] = True
    if self.at_roof:
      state[0, self.helicopter_column, 2] = True
      state[0, self.helicopter_column, 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_hazards:
      state[*position, 4] = True
    if self.building >= 3:
      state[3, 7:9, 1] = True
      state[3, 7:9, 3] = True
    state[9, : self.lives, 5] = True
    state[8, : self.building, 5] = True
    if self.reaching:
      state[7, :2, 5] = True
    state[6, : min(10, self.electric_shocks), 5] = True
    return state
