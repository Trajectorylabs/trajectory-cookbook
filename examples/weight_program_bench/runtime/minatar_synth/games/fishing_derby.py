import numpy as np


class FishingDerbyEnvironment:
  """Race-to-99 fishing duel with depth values and shark interception."""

  score_cap = 99.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 600):
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
    self.player_line = [2, 1]
    self.computer_line = [2, 8]
    self.previous_player_line = self.player_line.copy()
    self.previous_fish: list[list[int]] = []
    self.previous_hook_positions: dict[str, list[int]] = {}
    self.player_score = 0
    self.computer_score = 0
    self.player_hook: dict[str, object] | None = None
    self.computer_hook: dict[str, object] | None = None
    self.fish = []
    for index, row in enumerate(range(2, 8)):
      origin = "left" if index % 2 == 0 else "right"
      self.fish.append(
        {
          "position": [row, 0 if origin == "left" else 9],
          "direction": 1 if origin == "left" else -1,
          "origin": origin,
          "spawn_row": row,
          "value": 2 if row <= 3 else 4 if row <= 5 else 6,
          "hooked": None,
        }
      )
    self.shark = [1, 5]
    self.shark_direction = 1
    self.steps = 0
    self.last_action = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player_line = self.player_line.copy()
    self.previous_fish = [list(fish["position"]) for fish in self.fish]
    self.previous_hook_positions = {}
    if self.player_hook is not None:
      self.previous_hook_positions["player"] = list(self.player_hook["position"])
    if self.computer_hook is not None:
      self.previous_hook_positions["computer"] = list(self.computer_hook["position"])
    self._move_player_line(action)
    self._move_computer_line()
    self._move_free_fish()
    self._move_shark()
    self._hook_fish()
    reward = self._resolve_catches()
    self.steps += 1
    if self.steps >= self.horizon or self.player_score >= 99 or self.computer_score >= 99:
      self.terminal = True
    return reward, self.terminal

  def _move_player_line(self, action: int) -> None:
    if self.player_hook is None:
      if action == 1:
        self.player_line[1] = max(0, self.player_line[1] - 1)
      elif action == 2:
        self.player_line[0] = max(2, self.player_line[0] - 1)
      elif action == 3:
        self.player_line[1] = min(9, self.player_line[1] + 1)
      elif action == 4:
        self.player_line[0] = min(7, self.player_line[0] + 1)
      return
    self._move_hooked_line(self.player_line, self.player_hook, action)

  def _move_computer_line(self) -> None:
    if self.computer_hook is not None:
      action = 5
      if self.computer_hook["position"][0] <= 2 and self.computer_hook["position"][1] == self.shark[1]:
        action = 1 if self.computer_line[1] > 0 else 3
      self._move_hooked_line(self.computer_line, self.computer_hook, action)
      return
    if self.player_hook is not None:
      return
    available = [fish for fish in self.fish if fish["hooked"] is None]
    if not available:
      return
    target = max(available, key=lambda fish: (int(fish["value"]), -abs(fish["position"][1] - 8)))
    if self.computer_line[0] < target["position"][0]:
      self.computer_line[0] += 1
    elif self.computer_line[0] > target["position"][0]:
      self.computer_line[0] -= 1
    elif self.computer_line[1] < target["position"][1]:
      self.computer_line[1] += 1
    elif self.computer_line[1] > target["position"][1]:
      self.computer_line[1] -= 1

  @staticmethod
  def _move_hooked_line(line: list[int], fish: dict[str, object], action: int) -> None:
    line[0] -= 2 if action == 5 else 1
    if action == 1:
      line[1] = max(0, line[1] - 1)
    elif action == 2:
      line[0] -= 1
    elif action == 3:
      line[1] = min(9, line[1] + 1)
    elif action == 4:
      line[0] += 1
    fish["position"] = line.copy()

  def _move_free_fish(self) -> None:
    for fish in self.fish:
      if fish["hooked"] is not None:
        continue
      fish["position"][1] += int(fish["direction"])
      if fish["position"][1] < 0 or fish["position"][1] > 9:
        fish["position"][1] %= 10

  def _move_shark(self) -> None:
    if self.random.rand() < 0.15:
      self.shark_direction *= -1
    self.shark[1] += self.shark_direction
    if self.shark[1] <= 0 or self.shark[1] >= 9:
      self.shark_direction *= -1
      self.shark[1] = int(np.clip(self.shark[1], 0, 9))

  def _hook_fish(self) -> None:
    if self.player_hook is not None or self.computer_hook is not None:
      return
    player_fish = next(
      (fish for fish in self.fish if fish["hooked"] is None and fish["position"] == self.player_line),
      None,
    )
    if player_fish is not None:
      player_fish["hooked"] = "player"
      self.player_hook = player_fish
      return
    computer_fish = next(
      (fish for fish in self.fish if fish["hooked"] is None and fish["position"] == self.computer_line),
      None,
    )
    if computer_fish is not None:
      computer_fish["hooked"] = "computer"
      self.computer_hook = computer_fish

  def _resolve_catches(self) -> float:
    reward = 0.0
    for owner in ("player", "computer"):
      fish = self.player_hook if owner == "player" else self.computer_hook
      if fish is None:
        continue
      previous = self.previous_hook_positions.get(owner, list(fish["position"]))
      crossed_shark = (
        previous[0] >= self.shark[0] >= fish["position"][0]
        and self.shark[1] in (previous[1], fish["position"][1])
      )
      if fish["position"] == self.shark or crossed_shark:
        self._respawn_fish(fish)
        if owner == "player":
          self.player_hook = None
        else:
          self.computer_hook = None
        continue
      if fish["position"][0] > 0:
        continue
      value = int(fish["value"])
      if owner == "player":
        awarded = min(value, 99 - self.player_score)
        self.player_score += awarded
        reward += float(awarded)
        self.player_hook = None
        self.player_line = [2, 1]
      else:
        self.computer_score = min(99, self.computer_score + value)
        self.computer_hook = None
        self.computer_line = [2, 8]
      self._respawn_fish(fish)
    return reward

  @staticmethod
  def _respawn_fish(fish: dict[str, object]) -> None:
    fish["position"] = [int(fish["spawn_row"]), 0 if fish["origin"] == "left" else 9]
    fish["direction"] = 1 if fish["origin"] == "left" else -1
    fish["hooked"] = None

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player_line, 0] = True
    state[*self.shark, 1] = True
    for fish in self.fish:
      state[*fish["position"], 2] = True
      if fish["hooked"] is not None:
        state[*fish["position"], 3] = True
    state[*self.computer_line, 3] = True
    state[*self.previous_player_line, 4] = True
    for position in self.previous_fish:
      if 0 <= position[0] < 10:
        state[*position, 4] = True
    for bit in range(7):
      if self.player_score & (1 << bit):
        state[9, bit, 5] = True
      if self.computer_score & (1 << bit):
        state[8, bit, 5] = True
    if self.player_hook is not None:
      state[7, : int(self.player_hook["value"]), 5] = True
    return state
