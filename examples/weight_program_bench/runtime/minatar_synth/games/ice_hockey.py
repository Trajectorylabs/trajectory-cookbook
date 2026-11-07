import numpy as np


class IceHockeyEnvironment:
  """Two-on-two timed hockey with automatic nearest-player control."""

  score_cap = 40.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 360):
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
    self.player_team = [[5, 3], [5, 1]]
    self.opponent_team = [[4, 6], [4, 8]]
    self.previous_player_team = [player.copy() for player in self.player_team]
    self.previous_opponent_team = [player.copy() for player in self.opponent_team]
    self.puck = [5, 5]
    self.previous_puck = self.puck.copy()
    self.puck_velocity = [0, 0]
    self.possession: tuple[str, int] | None = None
    self.player_stun = [0, 0]
    self.opponent_stun = [0, 0]
    self.player_score = 0
    self.opponent_score = 0
    self.clock = 0
    self.faceoff_timer = 3
    self.stick_phase = -1
    self.last_action = 0
    self.terminal = False

  def active_player(self, team: str) -> int:
    players = self.player_team if team == "player" else self.opponent_team
    return min(
      range(2),
      key=lambda index: (abs(players[index][0] - self.puck[0]) + abs(players[index][1] - self.puck[1]), index),
    )

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player_team = [player.copy() for player in self.player_team]
    self.previous_opponent_team = [player.copy() for player in self.opponent_team]
    self.previous_puck = self.puck.copy()
    reward = 0.0
    if self.faceoff_timer > 0:
      self.faceoff_timer -= 1
      if self.faceoff_timer == 0:
        self._capture_loose_puck()
    else:
      self._move_player(action)
      self._move_opponent()
      self._capture_loose_puck()
      reward = self._move_puck()
      self.stick_phase = -1 if self.stick_phase == 1 else self.stick_phase + 1
    self.player_stun = [max(0, value - 1) for value in self.player_stun]
    self.opponent_stun = [max(0, value - 1) for value in self.opponent_stun]
    self.clock += 1
    if self.clock >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    index = self.active_player("player")
    if self.player_stun[index] > 0:
      return
    if action in (1, 2, 3, 4):
      delta = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}[action]
      self._move_skater(self.player_team[index], delta)
      if self.possession == ("player", index):
        self.puck = self.player_team[index].copy()
    elif action == 5:
      if self.possession == ("player", index):
        self.possession = None
        self.puck_velocity = [self.stick_phase, 1]
      else:
        self._check("player", index)

  def _move_opponent(self) -> None:
    index = self.active_player("opponent")
    if self.opponent_stun[index] > 0:
      return
    skater = self.opponent_team[index]
    if self.possession == ("opponent", index) and skater[1] <= 3:
      self.possession = None
      self.puck_velocity = [self.stick_phase, -1]
      return
    target = [5, 0] if self.possession == ("opponent", index) else self.puck
    row_delta = int(np.sign(target[0] - skater[0]))
    column_delta = int(np.sign(target[1] - skater[1]))
    delta = (0, column_delta) if column_delta else (row_delta, 0)
    self._move_skater(skater, delta)
    if self.possession == ("opponent", index):
      self.puck = skater.copy()
    elif any(abs(skater[0] - player[0]) + abs(skater[1] - player[1]) <= 1 for player in self.player_team):
      self._check("opponent", index)

  @staticmethod
  def _move_skater(skater: list[int], delta: tuple[int, int]) -> None:
    skater[0] = int(np.clip(skater[0] + delta[0], 0, 9))
    skater[1] = int(np.clip(skater[1] + delta[1], 0, 9))

  def _check(self, team: str, index: int) -> None:
    checker = self.player_team[index] if team == "player" else self.opponent_team[index]
    other_team = self.opponent_team if team == "player" else self.player_team
    other_stun = self.opponent_stun if team == "player" else self.player_stun
    for other_index, skater in enumerate(other_team):
      if abs(checker[0] - skater[0]) + abs(checker[1] - skater[1]) <= 1:
        other_stun[other_index] = 2
        if self.possession == (("opponent" if team == "player" else "player"), other_index):
          self.possession = None
          self.puck = skater.copy()
          self.puck_velocity = [0, 0]
        return

  def _capture_loose_puck(self) -> None:
    if self.possession is not None or self.puck_velocity != [0, 0]:
      return
    candidates = []
    for team, players, stuns in (
      ("player", self.player_team, self.player_stun),
      ("opponent", self.opponent_team, self.opponent_stun),
    ):
      for index, skater in enumerate(players):
        distance = abs(skater[0] - self.puck[0]) + abs(skater[1] - self.puck[1])
        if distance <= 1 and stuns[index] == 0:
          candidates.append((distance, 0 if team == "player" else 1, team, index))
    if candidates:
      _, _, team, index = min(candidates)
      self.possession = (team, index)
      players = self.player_team if team == "player" else self.opponent_team
      self.puck = players[index].copy()

  def _move_puck(self) -> float:
    if self.possession is not None or self.puck_velocity == [0, 0]:
      return 0.0
    self.puck[0] += self.puck_velocity[0]
    self.puck[1] += self.puck_velocity[1]
    if self.puck[0] < 0 or self.puck[0] > 9:
      self.puck[0] = int(np.clip(self.puck[0], 0, 9))
      self.puck_velocity[0] *= -1
    if self.puck[1] >= 9:
      if self.puck[0] in (4, 5):
        self.player_score += 1
        self._faceoff()
        return 1.0
      self.puck[1] = 9
      self.puck_velocity[1] = -1
    elif self.puck[1] <= 0:
      if self.puck[0] in (4, 5):
        self.opponent_score += 1
        self._faceoff()
        return 0.0
      self.puck[1] = 0
      self.puck_velocity[1] = 1
    for team, players in (("player", self.player_team), ("opponent", self.opponent_team)):
      for index, skater in enumerate(players):
        if skater == self.puck:
          self.possession = (team, index)
          self.puck_velocity = [0, 0]
          return 0.0
    return 0.0

  def _faceoff(self) -> None:
    self.player_team = [[5, 3], [5, 1]]
    self.opponent_team = [[4, 6], [4, 8]]
    self.puck = [5, 5]
    self.puck_velocity = [0, 0]
    self.possession = None
    self.faceoff_timer = 3

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for skater in self.player_team:
      state[*skater, 0] = True
    for skater in self.opponent_team:
      state[*skater, 1] = True
    state[*self.puck, 2] = True
    if self.possession is not None:
      state[*self.puck, 3] = True
    for skater in self.previous_player_team + self.previous_opponent_team:
      state[*skater, 4] = True
    state[*self.previous_puck, 4] = True
    for bit in range(6):
      state[9, bit, 5] = bool(self.player_score & (1 << bit))
      state[8, bit, 5] = bool(self.opponent_score & (1 << bit))
    for bit in range(9):
      state[7, bit, 5] = bool(self.clock & (1 << bit))
    state[6, self.active_player("player"), 5] = True
    state[6, 2 + self.active_player("opponent"), 5] = True
    state[6, 4 + self.puck_velocity[0], 5] = True
    state[5, 4 + self.puck_velocity[1], 5] = True
    state[5, 7 + self.stick_phase, 5] = True
    if self.possession is not None:
      state[4, 0 if self.possession[0] == "player" else 1, 5] = True
      state[4, 2 + self.possession[1], 5] = True
    for index, stun in enumerate(self.player_stun):
      state[3, index, 5] = stun > 0
    for index, stun in enumerate(self.opponent_stun):
      state[3, 2 + index, 5] = stun > 0
    state[3, 4 : 4 + self.faceoff_timer, 5] = True
    return state
