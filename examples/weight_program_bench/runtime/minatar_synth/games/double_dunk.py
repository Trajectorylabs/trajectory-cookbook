import numpy as np


BASKET = [1, 5]
MOVE_DELTAS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


class DoubleDunkEnvironment:
  """First-to-24 two-on-two half-court basketball game."""

  score_cap = 24.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 300):
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
    self.player_team = [[7, 3], [4, 5]]
    self.opponent_team = [[7, 7], [3, 5]]
    self.previous_positions = [position.copy() for position in self.player_team + self.opponent_team]
    self.possession = "player"
    self.ball_handler = 0
    self.player_score = 0
    self.opponent_score = 0
    self.phase = "live"
    self.jump_age = 0
    self.pass_used = False
    self.shot_clock = 10
    self.lane_steps = 0
    self.cleared = True
    self.block_attempt = False
    self.steps = 0
    self.last_action = 0
    self.last_ball_position: list[int] | None = None
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_positions = [position.copy() for position in self.player_team + self.opponent_team]
    self.last_ball_position = self._ball_position().copy()
    reward = 0.0
    if self.possession == "player":
      reward += self._player_offense(action)
      self._computer_defense()
    else:
      self._player_defense(action)
      self._computer_offense()
    if self.phase == "jumping":
      self.jump_age += 1
    self._update_violations()
    self.steps += 1
    if self.steps >= self.horizon or self.player_score >= 24 or self.opponent_score >= 24:
      self.terminal = True
    return reward, self.terminal

  def _player_offense(self, action: int) -> float:
    handler = self.player_team[self.ball_handler]
    if self.phase == "jumping":
      if action == 5:
        return self._resolve_shot("player")
      return 0.0
    if action in MOVE_DELTAS:
      self._move(handler, action)
      if not self.cleared and self._outside_arc(handler):
        self.cleared = True
    elif action == 5:
      if not self.pass_used and self._distance(handler, BASKET) > 1:
        self.ball_handler = 1 - self.ball_handler
        self.pass_used = True
      else:
        self.phase = "jumping"
        self.jump_age = 0
    return 0.0

  def _player_defense(self, action: int) -> None:
    defender = self.player_team[self.ball_handler]
    if action in MOVE_DELTAS:
      self._move(defender, action)
    elif action == 5:
      if self.phase == "jumping":
        self.block_attempt = True
      else:
        self._attempt_steal()

  def _computer_defense(self) -> None:
    handler = self.player_team[self.ball_handler]
    defender = self.opponent_team[self.ball_handler]
    if self.phase == "jumping":
      return
    self._step_toward(defender, handler)

  def _computer_offense(self) -> None:
    handler = self.opponent_team[self.ball_handler]
    defender = self.player_team[self.ball_handler]
    if self.phase == "jumping":
      if self.jump_age >= 1:
        self._resolve_shot("opponent")
      return
    if not self.cleared:
      target = [7, 2 if handler[1] < 5 else 8]
      self._step_toward(handler, target)
      if self._outside_arc(handler):
        self.cleared = True
      return
    distance = self._distance(handler, BASKET)
    if distance <= 2 or self.shot_clock <= 3:
      self.phase = "jumping"
      self.jump_age = 0
    else:
      self._step_toward(handler, BASKET)
    if self.phase == "live" and self._distance(handler, defender) == 0:
      handler[1] = int(np.clip(handler[1] + (1 if handler[1] <= 5 else -1), 0, 9))

  def _resolve_shot(self, team: str) -> float:
    offense = self.player_team if team == "player" else self.opponent_team
    defense = self.opponent_team if team == "player" else self.player_team
    handler = offense[self.ball_handler]
    defender = defense[self.ball_handler]
    distance = self._distance(handler, BASKET)
    if not self.cleared:
      self._change_possession()
      return 0.0
    points = 3 if self._outside_arc(handler) else 2
    dunk = distance <= 1
    skill = 0.82 if (self.ball_handler == 1 and distance <= 3) else 0.72 if (
      self.ball_handler == 0 and self._outside_arc(handler)
    ) else 0.58
    if dunk:
      skill = 0.96
    if self.jump_age > 2:
      skill -= 0.35
    guard_distance = self._distance(handler, defender)
    if guard_distance <= 1:
      skill -= 0.25
    if self.block_attempt and guard_distance <= 2:
      skill -= 0.4
    made = self.random.rand() < max(0.05, skill)
    self.phase = "live"
    self.jump_age = 0
    self.block_attempt = False
    if made:
      reward = 0.0
      if team == "player":
        awarded = min(points, 24 - self.player_score)
        self.player_score += awarded
        reward = float(awarded)
      else:
        self.opponent_score = min(24, self.opponent_score + points)
      self._change_possession()
      return reward
    self._resolve_rebound(team)
    return 0.0

  def _resolve_rebound(self, shooting_team: str) -> None:
    player_inside_distance = self._distance(self.player_team[1], BASKET)
    opponent_inside_distance = self._distance(self.opponent_team[1], BASKET)
    player_weight = 0.65 if player_inside_distance <= opponent_inside_distance else 0.35
    player_rebound = self.random.rand() < player_weight
    rebound_team = "player" if player_rebound else "opponent"
    self.possession = rebound_team
    self.ball_handler = 1
    self.phase = "live"
    self.pass_used = True
    self.shot_clock = 10
    self.lane_steps = 0
    self.cleared = rebound_team == shooting_team

  def _attempt_steal(self) -> None:
    ball_carrier = self.opponent_team[self.ball_handler]
    defender = self.player_team[self.ball_handler]
    distance = self._distance(ball_carrier, defender)
    if distance <= 1:
      chance = 0.6 if self.ball_handler == 0 else 0.35
      if self.random.rand() < chance:
        self._change_possession("player")
        return
    if distance > 1 and self.random.rand() < 0.25:
      self._free_throw("opponent")

  def _free_throw(self, team: str) -> None:
    made = self.random.rand() < 0.7
    if made and team == "player":
      self.player_score = min(24, self.player_score + 1)
    elif made:
      self.opponent_score = min(24, self.opponent_score + 1)
    self._change_possession("player" if team == "opponent" else "opponent")

  def _update_violations(self) -> None:
    if self.phase != "live":
      return
    self.shot_clock -= 1
    handler = self._ball_position()
    in_lane = handler[0] <= 4 and 4 <= handler[1] <= 6
    self.lane_steps = self.lane_steps + 1 if in_lane else 0
    if self.shot_clock <= 0 or self.lane_steps > 3:
      self._change_possession()

  def _change_possession(self, possession: str | None = None) -> None:
    self.possession = possession or ("opponent" if self.possession == "player" else "player")
    self.ball_handler = 0
    self.phase = "live"
    self.jump_age = 0
    self.pass_used = False
    self.shot_clock = 10
    self.lane_steps = 0
    self.cleared = True
    self.block_attempt = False

  def _ball_position(self) -> list[int]:
    team = self.player_team if self.possession == "player" else self.opponent_team
    return team[self.ball_handler]

  @staticmethod
  def _move(position: list[int], action: int) -> None:
    row_delta, column_delta = MOVE_DELTAS[action]
    position[0] = int(np.clip(position[0] + row_delta, 1, 9))
    position[1] = int(np.clip(position[1] + column_delta, 0, 9))

  def _step_toward(self, position: list[int], target: list[int]) -> None:
    if position[0] != target[0]:
      position[0] += int(np.sign(target[0] - position[0]))
    elif position[1] != target[1]:
      position[1] += int(np.sign(target[1] - position[1]))

  @staticmethod
  def _distance(first: list[int], second: list[int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])

  def _outside_arc(self, position: list[int]) -> bool:
    return self._distance(position, BASKET) >= 6

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for position in self.player_team:
      state[*position, 0] = True
    for position in self.opponent_team:
      state[*position, 1] = True
    state[*BASKET, 2] = True
    state[*self._ball_position(), 3] = True
    for position in self.previous_positions:
      state[*position, 4] = True
    state[9 - self.player_score // 10, self.player_score % 10, 5] = True
    state[6 - self.opponent_score // 10, self.opponent_score % 10, 5] = True
    state[3, : self.shot_clock, 5] = True
    state[2, : self.lane_steps, 5] = True
    if self.phase == "jumping":
      state[1, : min(10, self.jump_age + 1), 5] = True
    if not self.cleared:
      state[0, :3, 5] = True
    return state
