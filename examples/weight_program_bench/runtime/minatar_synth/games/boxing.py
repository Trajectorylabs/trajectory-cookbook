import numpy as np


DELTAS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


class BoxingEnvironment:
  """Two-minute top-down bout with guarded jabs, power punches, and counter windows."""

  score_cap = 100.0

  def __init__(self, sticky_action_prob: float = 0.1, round_steps: int = 120):
    self.sticky_action_prob = sticky_action_prob
    self.round_steps = round_steps
    self.random = np.random.RandomState()
    self.seed(0)
    self.reset()

  @classmethod
  def normalize_score(cls, raw_score: float) -> float:
    return min(max(raw_score, 0.0), cls.score_cap) / cls.score_cap

  def seed(self, seed: int) -> None:
    self.random.seed(seed)

  def reset(self) -> None:
    self.player = [5, 2]
    self.opponent = [5, 7]
    self.previous_player = self.player.copy()
    self.previous_opponent = self.opponent.copy()
    self.player_score = 0
    self.opponent_score = 0
    self.player_recovery = 0
    self.opponent_recovery = 0
    self.player_punch: list[int] | None = None
    self.opponent_punch: list[int] | None = None
    self.steps = 0
    self.last_action = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_opponent = self.opponent.copy()
    self.player_punch = None
    self.opponent_punch = None
    reward = 0.0
    if action in DELTAS:
      self._move(self.player, self.opponent, action)
    elif action == 5 and self.player_recovery == 0:
      scored = self._punch(self.player, self.opponent, defender_is_player=False)
      reward = float(min(scored, 100 - self.player_score))
      self.player_score += int(reward)
      self.player_recovery = 2
    if self.player_score >= 100:
      self.player_score = 100
      self.terminal = True
      return reward, True

    if self.steps < 60 or self.steps % 2 == 0:
      self._opponent_action()
    self.player_recovery = max(0, self.player_recovery - 1)
    self.opponent_recovery = max(0, self.opponent_recovery - 1)
    self.steps += 1
    if self.opponent_score >= 100 or self.steps >= self.round_steps:
      self.opponent_score = min(100, self.opponent_score)
      self.terminal = True
    return reward, self.terminal

  @staticmethod
  def _move(boxer: list[int], other: list[int], action: int) -> None:
    row_delta, column_delta = DELTAS[action]
    target = [boxer[0] + row_delta, boxer[1] + column_delta]
    if 1 <= target[0] <= 8 and 1 <= target[1] <= 8 and target != other:
      boxer[:] = target

  def _punch(self, attacker: list[int], defender: list[int], defender_is_player: bool) -> int:
    row_distance = abs(attacker[0] - defender[0])
    column_distance = abs(attacker[1] - defender[1])
    distance = max(row_distance, column_distance)
    if distance > 2:
      return 0
    recovery = self.player_recovery if defender_is_player else self.opponent_recovery
    angled = row_distance > 0 and column_distance > 0
    if not angled and recovery == 0:
      return 0
    points = 2 if distance == 1 else 1
    punch_cell = [
      attacker[0] + int(np.sign(defender[0] - attacker[0])),
      attacker[1] + int(np.sign(defender[1] - attacker[1])),
    ]
    if defender_is_player:
      self.opponent_punch = punch_cell
    else:
      self.player_punch = punch_cell
    self._push(attacker, defender)
    return points

  @staticmethod
  def _push(attacker: list[int], defender: list[int]) -> None:
    row_delta = int(np.sign(defender[0] - attacker[0]))
    column_delta = int(np.sign(defender[1] - attacker[1]))
    target = [
      int(np.clip(defender[0] + row_delta, 1, 8)),
      int(np.clip(defender[1] + column_delta, 1, 8)),
    ]
    if target != attacker:
      defender[:] = target

  def _opponent_action(self) -> None:
    row_distance = abs(self.opponent[0] - self.player[0])
    column_distance = abs(self.opponent[1] - self.player[1])
    if max(row_distance, column_distance) <= 2 and self.opponent_recovery == 0:
      self.opponent_score += self._punch(self.opponent, self.player, defender_is_player=True)
      self.opponent_recovery = 2
      return
    candidates = []
    for action in DELTAS:
      row_delta, column_delta = DELTAS[action]
      target = [self.opponent[0] + row_delta, self.opponent[1] + column_delta]
      if not (1 <= target[0] <= 8 and 1 <= target[1] <= 8) or target == self.player:
        continue
      distance = max(abs(target[0] - self.player[0]), abs(target[1] - self.player[1]))
      angled_bonus = -1 if target[0] != self.player[0] and target[1] != self.player[1] else 0
      candidates.append((distance + angled_bonus, self.random.rand(), action))
    if candidates:
      self._move(self.opponent, self.player, min(candidates)[2])

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    state[*self.opponent, 1] = True
    state[*self.opponent, 2] = True
    if self.player_punch is not None:
      state[*self.player_punch, 3] = True
    state[*self.previous_player, 4] = True
    state[*self.previous_opponent, 4] = True
    state[9, : min(10, (self.player_score + 9) // 10), 5] = True
    state[8, : min(10, (self.opponent_score + 9) // 10), 5] = True
    remaining = max(0, self.round_steps - self.steps)
    state[7, : min(10, (remaining + 11) // 12), 5] = True
    state[6, : self.player_recovery, 5] = True
    return state
