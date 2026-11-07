import numpy as np


TREASURE_VALUES = (2_000, 3_000, 4_000, 5_000)
TREASURE_SCENES = tuple(range(16)) + tuple(range(32, 48))


class PitfallEnvironment:
  """Mapped surface-and-tunnel treasure hunt across a 64-scene ring."""

  score_cap = 114_000.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 500):
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
    self.scene = 0
    self.player = [4, 1]
    self.previous_player = self.player.copy()
    self.level = 0
    self.lives = 3
    self.score = 2_000.0
    self.steps = 0
    self.last_action = 0
    self.jump_timer = 0
    self.swing_timer = 0
    self.terminal = False
    self.claimed: set[int] = set()
    self.log_column = 8
    self.scorpion_column = 8
    self.scorpion_direction = -1
    self.vine_column = 4
    self.vine_direction = 1
    self.previous_log_column = self.log_column
    self.previous_scorpion_column = self.scorpion_column
    self.previous_vine_column = self.vine_column

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_log_column = self.log_column
    self.previous_scorpion_column = self.scorpion_column
    self.previous_vine_column = self.vine_column
    scene_before = self.scene
    self._move_player(action)
    reward = self._collect_treasure()
    if self.scene == scene_before and not self.terminal:
      self._move_hazards()
      reward += self._resolve_hazards()
    self.score = max(0.0, self.score + reward)
    self.steps += 1
    self.jump_timer = max(0, self.jump_timer - 1)
    self.swing_timer = max(0, self.swing_timer - 1)
    if self.steps >= self.horizon or len(self.claimed) == 32:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    if self.swing_timer:
      self.player[1] = self.vine_column
      if action == 4:
        self.swing_timer = 0
        self.jump_timer = 2
      return
    if action == 5:
      if self.level == 0 and abs(self.player[1] - self.vine_column) <= 1 and self._hazard_kind() == "pit":
        self.swing_timer = 4
        self.player[1] = self.vine_column
      else:
        self.jump_timer = 3
        self.player[0] = 2 if self.level == 0 else 6
      return
    if action == 1:
      self.player[1] -= 1
    elif action == 3:
      self.player[1] += 1
    elif action == 4 and self.level == 0 and self._has_ladder() and self.player[1] == 5:
      self.level = 1
      self.player = [8, 5]
    elif action == 2 and self.level == 1 and self._has_ladder() and self.player[1] == 5:
      self.level = 0
      self.player = [4, 5]
    if self.jump_timer == 0:
      self.player[0] = 4 if self.level == 0 else 8
    if self.player[1] < 0:
      self._change_scene(-1)
    elif self.player[1] > 9:
      self._change_scene(1)

  def _change_scene(self, direction: int) -> None:
    distance = 1 if self.level == 0 else 3
    self.scene = (self.scene + direction * distance) % 64
    self.player = [4 if self.level == 0 else 8, 9 if direction < 0 else 0]
    self.previous_player = self.player.copy()
    self.log_column = 8
    self.scorpion_column = 8
    self.scorpion_direction = -1
    self.vine_column = 4
    self.vine_direction = 1
    self.previous_log_column = self.log_column
    self.previous_scorpion_column = self.scorpion_column
    self.previous_vine_column = self.vine_column

  def _has_ladder(self) -> bool:
    return self.scene % 4 == 0

  def _hazard_kind(self) -> str:
    kinds = ("logs", "pit", "crocodiles", "snake", "fire", "hole", "clear")
    return kinds[self.scene % len(kinds)]

  def _collect_treasure(self) -> float:
    if self.level or self.scene not in TREASURE_SCENES or self.scene in self.claimed:
      return 0.0
    if self.player != [4, 8]:
      return 0.0
    self.claimed.add(self.scene)
    treasure_index = TREASURE_SCENES.index(self.scene)
    return float(TREASURE_VALUES[treasure_index % 4])

  def _move_hazards(self) -> None:
    if self.level == 0:
      if self._hazard_kind() == "logs":
        self.log_column -= 1
        if self.log_column < 0:
          self.log_column = 9
      if self._hazard_kind() == "pit":
        self.vine_column += self.vine_direction
        if self.vine_column in (3, 7):
          self.vine_direction *= -1
    else:
      self.scorpion_column += self.scorpion_direction
      if self.scorpion_column in (1, 8):
        self.scorpion_direction *= -1

  def _resolve_hazards(self) -> float:
    if self.level == 1:
      if self.player == [8, self.scorpion_column] and self.jump_timer == 0:
        self._lose_life()
      return 0.0
    kind = self._hazard_kind()
    if kind == "logs" and self.player[1] == self.log_column and self.jump_timer == 0:
      return -10.0
    if kind == "hole" and self.player[1] in (4, 5) and self.jump_timer == 0:
      self.level = 1
      self.player = [8, 5]
      return -100.0
    dangerous = (
      kind == "pit" and self.player[1] in (4, 5, 6) and self.jump_timer == 0 and self.swing_timer == 0
    ) or (kind == "crocodiles" and self.player[1] in (4, 5, 6) and self.jump_timer == 0) or (
      kind in ("snake", "fire") and self.player[1] == 6 and self.jump_timer == 0
    )
    if dangerous:
      self._lose_life()
    return 0.0

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    self.player = [4 if self.level == 0 else 8, 1]
    self.previous_player = self.player.copy()
    self.jump_timer = 0
    self.swing_timer = 0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    state[*self.previous_player, 4] = True
    if self.scene in TREASURE_SCENES and self.scene not in self.claimed:
      state[4, 8, 2] = True
    state[4, :, 3] = True
    state[8, :, 3] = True
    if self._has_ladder():
      state[4:9, 5, 3] = True
    kind = self._hazard_kind()
    if self.level == 0:
      if kind == "logs":
        state[4, self.log_column, 1] = True
        state[4, self.previous_log_column, 4] = True
      elif kind in ("pit", "crocodiles", "hole"):
        state[4, 4:7, 3] = False
        state[5, 4:7, 1] = True
      elif kind in ("snake", "fire"):
        state[4, 6, 1] = True
      if kind == "pit":
        state[1:5, self.vine_column, 3] = True
        state[1:5, self.previous_vine_column, 4] = True
    else:
      state[8, self.scorpion_column, 1] = True
      state[8, self.previous_scorpion_column, 4] = True
    for index, scene in enumerate(TREASURE_SCENES):
      state[index // 10, index % 10, 5] = scene in self.claimed
    for bit in range(6):
      index = 32 + bit
      state[index // 10, index % 10, 5] = bool(self.scene & (1 << bit))
    state[3, 8, 5] = bool(self.level)
    for bit in range(2):
      index = 39 + bit
      state[index // 10, index % 10, 5] = bool(self.lives & (1 << bit))
    for bit in range(9):
      index = 41 + bit
      state[index // 10, index % 10, 5] = bool(self.steps & (1 << bit))
    for bit in range(2):
      state[5, bit, 5] = bool(self.jump_timer & (1 << bit))
      state[5, 2 + bit, 5] = bool(self.swing_timer & (1 << bit))
    state[5, 4, 5] = self.scorpion_direction > 0
    state[5, 5, 5] = self.vine_direction > 0
    return state
