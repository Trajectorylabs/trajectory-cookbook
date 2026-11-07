import numpy as np


CARROT_COLUMNS = (3, 5, 7)


class GopherEnvironment:
  """Twelve-raid carrot defense with tunnel repair and seed recovery."""

  score_cap = 4_560.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_200):
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
    self.farmer = [2, 5]
    self.previous_farmer = self.farmer.copy()
    self.carrots = set(CARROT_COLUMNS)
    self.carrying_seed = False
    self.seed_position: list[int] | None = None
    self.duck: list[int] | None = None
    self.raid = 0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.previous_gopher: list[int] | None = None
    self._start_raid()

  def _start_raid(self) -> None:
    if self.raid >= 12 or not self.carrots:
      self.terminal = True
      return
    target_column = sorted(self.carrots)[self.raid % len(self.carrots)]
    start_column = 0 if self.raid % 2 == 0 else 9
    step = 1 if target_column > start_column else -1
    horizontal = [[8, column] for column in range(start_column, target_column + step, step)]
    vertical = [[row, target_column] for row in range(7, 1, -1)]
    self.gopher_path = horizontal + vertical
    self.gopher_index = 0
    self.gopher = self.gopher_path[0].copy()
    self.tunnels: set[tuple[int, int]] = {tuple(self.gopher)}
    self.scored_sections: set[tuple[int, int]] = set()
    self.raid_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_farmer = self.farmer.copy()
    self.previous_gopher = self.gopher.copy()
    raid_before_action = self.raid
    reward = self._apply_action(action)
    if self.raid == raid_before_action and not self.terminal:
      self._move_gopher()
    self._move_duck_and_seed()
    self._resolve_seed_catch()
    self.raid_steps += 1
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _apply_action(self, action: int) -> float:
    if action == 1:
      self.farmer[1] = max(0, self.farmer[1] - 1)
    elif action == 3:
      self.farmer[1] = min(9, self.farmer[1] + 1)
    elif action == 5:
      if self.carrying_seed and self.farmer[1] in CARROT_COLUMNS and self.farmer[1] not in self.carrots:
        self.carrots.add(self.farmer[1])
        self.carrying_seed = False
        return 0.0
      if self.gopher[0] == 2 and self.gopher[1] == self.farmer[1]:
        self.raid += 1
        self._start_raid()
        return 100.0
      column_sections = [section for section in self.tunnels if section[1] == self.farmer[1]]
      if column_sections:
        section = min(column_sections)
        self.tunnels.remove(section)
        self.gopher_index = max(0, self.gopher_index - 1)
        self.gopher = self.gopher_path[self.gopher_index].copy()
        if section not in self.scored_sections:
          self.scored_sections.add(section)
          return 20.0
    return 0.0

  def _move_gopher(self) -> None:
    interval = max(1, 5 - self.raid // 3)
    if self.raid_steps % interval:
      return
    if self.gopher_index < len(self.gopher_path) - 1:
      self.gopher_index += 1
      self.gopher = self.gopher_path[self.gopher_index].copy()
      self.tunnels.add(tuple(self.gopher))
      return
    target_column = self.gopher[1]
    self.carrots.discard(target_column)
    self.raid += 1
    self._start_raid()

  def _move_duck_and_seed(self) -> None:
    missing = [column for column in CARROT_COLUMNS if column not in self.carrots]
    if missing and self.duck is None and self.seed_position is None and not self.carrying_seed:
      if self.steps % 24 == 0:
        self.duck = [0, 0]
    if self.duck is not None:
      self.duck[1] += 1
      if self.duck[1] == 5 and self.seed_position is None:
        self.seed_position = [0, 5]
      if self.duck[1] > 9:
        self.duck = None
    if self.seed_position is not None:
      self.seed_position[0] += 1
      if self.seed_position[0] > 2:
        self.seed_position = None

  def _resolve_seed_catch(self) -> None:
    if self.seed_position == self.farmer:
      self.seed_position = None
      self.carrying_seed = True

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.farmer, 0] = True
    state[*self.gopher, 1] = True
    for column in self.carrots:
      state[1, column, 2] = True
    if self.seed_position is not None:
      state[*self.seed_position, 2] = True
    if self.duck is not None:
      state[*self.duck, 2] = True
    for section in self.tunnels:
      state[*section, 3] = True
    state[*self.previous_farmer, 4] = True
    if self.previous_gopher is not None:
      state[*self.previous_gopher, 4] = True
    state[9, : len(self.carrots), 5] = True
    for bit in range(4):
      state[8, bit, 5] = bool(self.raid & (1 << bit))
    for bit in range(4):
      state[7, bit, 5] = bool(len(self.tunnels) & (1 << bit))
    if self.carrying_seed:
      state[6, :3, 5] = True
    return state
