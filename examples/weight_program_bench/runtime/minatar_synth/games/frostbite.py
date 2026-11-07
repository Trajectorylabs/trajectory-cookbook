import numpy as np


ICE_ROWS = (3, 4, 5, 6)


class FrostbiteEnvironment:
  """Nine-level ice-floe construction game with temperature pressure."""

  score_cap = 41_850.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 2_025):
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
    self.player = [7, 5]
    self.previous_player = self.player.copy()
    self.level = 1
    self.lives = 4
    self.score = 0.0
    self.next_life_score = 5_000.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.life_lost_this_step = False
    self._start_level()

  def _start_level(self) -> None:
    self.player = [7, 5]
    self.previous_player = self.player.copy()
    self.life_lost_this_step = False
    self.floes = {row: {(row + offset * 3) % 10 for offset in range(4)} for row in ICE_ROWS}
    self.floe_directions = {row: 1 if row % 2 else -1 for row in ICE_ROWS}
    self.white_rows = set(ICE_ROWS)
    self.igloo_blocks = 0
    self.scored_blocks = 0
    self.temperature = 45
    self.level_steps = 0
    self.fish_remaining = 4
    self.fish: list[dict[str, object]] = []
    self.hazards = [
      {
        "position": [row, (row * 2 + self.level) % 10],
        "direction": -self.floe_directions[row],
        "kind": ("clam", "crab", "goose")[(row - 3) % 3],
      }
      for row in ICE_ROWS
    ]
    self.bear = [2, 9] if self.level >= 4 else None
    self.previous_objects: list[list[int]] = []

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_objects = [list(hazard["position"]) for hazard in self.hazards]
    self.previous_objects.extend(list(fish["position"]) for fish in self.fish)
    level_before_action = self.level
    reward = self._apply_action(action)
    if self.level != level_before_action or self.terminal:
      self.score += reward
      self.steps += 1
      if self.steps >= self.horizon:
        self.terminal = True
      return reward, self.terminal
    self._move_world()
    reward += self._resolve_landing()
    reward += self._resolve_fish()
    self._resolve_hazards()
    self.level_steps += 1
    self.steps += 1
    if self.level_steps % 5 == 0 and not self.life_lost_this_step:
      self.temperature -= 1
      if self.temperature <= 0:
        self._lose_life()
    self.score += reward
    while self.score >= self.next_life_score and self.lives < 10:
      self.lives += 1
      self.next_life_score += 5_000.0
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _apply_action(self, action: int) -> float:
    if action == 5:
      for row in ICE_ROWS:
        self.floe_directions[row] *= -1
      if self.igloo_blocks < 16 and self.igloo_blocks > 0:
        self.igloo_blocks -= 1
      return 0.0
    if action in (1, 2, 3, 4):
      row_delta, column_delta = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}[action]
      target = [self.player[0] + row_delta, self.player[1] + column_delta]
      if 1 <= target[0] <= 7 and 0 <= target[1] < 10:
        self.player = target
    if self.igloo_blocks >= 16 and self.player == [1, 9]:
      return self._enter_igloo()
    return 0.0

  def _move_world(self) -> None:
    standing_row = self.previous_player[0]
    carried = standing_row in ICE_ROWS and self.previous_player[1] in self.floes[standing_row]
    for row in ICE_ROWS:
      direction = self.floe_directions[row]
      self.floes[row] = {(column + direction) % 10 for column in self.floes[row]}
    if carried and self.player == self.previous_player:
      self.player[1] = (self.player[1] + self.floe_directions[standing_row]) % 10
    for hazard in self.hazards:
      hazard["position"][1] = (hazard["position"][1] + int(hazard["direction"])) % 10
    for fish in self.fish:
      fish["position"][1] = (fish["position"][1] + int(fish["direction"])) % 10
    if self.fish_remaining > 0 and self.level_steps % 12 == 0:
      row = ICE_ROWS[self.fish_remaining % 4]
      self.fish.append(
        {"position": [row, int(self.random.randint(10))], "direction": self.floe_directions[row]}
      )
      self.fish_remaining -= 1
    if self.bear is not None:
      self.bear[1] += int(np.sign(self.player[1] - self.bear[1]))

  def _resolve_landing(self) -> float:
    row, column = self.player
    if row in ICE_ROWS and column not in self.floes[row]:
      self._lose_life()
      return 0.0
    if row not in self.white_rows or row not in ICE_ROWS:
      return 0.0
    self.white_rows.remove(row)
    self.igloo_blocks += 1
    reward = 0.0
    if self.scored_blocks < 16:
      self.scored_blocks += 1
      reward = float(min(self.level, 9) * 10)
    if not self.white_rows and self.igloo_blocks < 16:
      self.white_rows = set(ICE_ROWS)
    return reward

  def _resolve_fish(self) -> float:
    caught = [fish for fish in self.fish if fish["position"] == self.player]
    if not caught:
      return 0.0
    caught_ids = {id(fish) for fish in caught}
    self.fish = [fish for fish in self.fish if id(fish) not in caught_ids]
    return 200.0 * len(caught)

  def _resolve_hazards(self) -> None:
    if self.life_lost_this_step:
      return
    if any(hazard["position"] == self.player for hazard in self.hazards):
      self._lose_life()
      return
    if self.bear is not None and self.bear == self.player:
      self._lose_life()

  def _lose_life(self) -> None:
    if self.life_lost_this_step:
      return
    self.life_lost_this_step = True
    self.lives -= 1
    self.player = [7, 5]
    self.previous_player = self.player.copy()
    self.temperature = 45
    if self.lives <= 0:
      self.terminal = True

  def _enter_igloo(self) -> float:
    level_value = min(self.level, 9)
    reward = float(160 * level_value + 10 * self.temperature * level_value)
    self.level += 1
    if self.level > 9:
      self.terminal = True
    else:
      self._start_level()
    return reward

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for row in ICE_ROWS:
      for column in self.floes[row]:
        state[row, column, 2] = True
        if row in self.white_rows:
          state[row, column, 3] = True
    for hazard in self.hazards:
      state[*hazard["position"], 1] = True
    if self.bear is not None:
      state[*self.bear, 1] = True
    for fish in self.fish:
      state[*fish["position"], 2] = True
    if self.igloo_blocks >= 16:
      state[1, 9, 2] = True
      state[1, 9, 3] = True
    state[*self.previous_player, 4] = True
    for position in self.previous_objects:
      state[*position, 4] = True
    state[9, : self.lives, 5] = True
    for bit in range(5):
      if self.igloo_blocks & (1 << bit):
        state[8, bit, 5] = True
    state[7, : self.level, 5] = True
    for bit in range(6):
      if self.temperature & (1 << bit):
        state[2, bit, 5] = True
    return state
