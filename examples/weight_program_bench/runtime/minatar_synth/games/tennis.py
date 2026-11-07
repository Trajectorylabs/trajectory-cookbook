import numpy as np


MOVE_DELTAS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}


class TennisEnvironment:
  """One Activision-style set with automatic positional racket contact."""

  score_cap = 6.0

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
    self.player = [8, 5]
    self.opponent = [1, 5]
    self.previous_player = self.player.copy()
    self.previous_opponent = self.opponent.copy()
    self.ball = self.player.copy()
    self.previous_ball = self.ball.copy()
    self.ball_velocity = [0, 0]
    self.server = "player"
    self.phase = "serve"
    self.serve_delay = 2
    self.player_points = 0
    self.opponent_points = 0
    self.player_games = 0
    self.opponent_games = 0
    self.tiebreak_reset = False
    self.side_swapped = False
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.winner: str | None = None
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    self.previous_opponent = self.opponent.copy()
    self.previous_ball = self.ball.copy()
    previous_progress = self._progress_score()
    self._move_player(action)
    if self.phase == "serve":
      self._serve(action)
    else:
      self._player_return()
      self._move_opponent()
      self._opponent_return()
      self._move_ball()
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    progress = self._progress_score()
    self.score = progress
    return progress - previous_progress, self.terminal

  def _move_player(self, action: int) -> None:
    if action not in MOVE_DELTAS:
      return
    row_delta, column_delta = MOVE_DELTAS[action]
    if self.phase == "serve":
      row_delta = 0
    self.player[0] = int(np.clip(self.player[0] + row_delta, 5, 9))
    self.player[1] = int(np.clip(self.player[1] + column_delta, 1, 8))
    if self.phase == "serve" and self.server == "player":
      self.ball = self.player.copy()

  def _serve(self, action: int) -> None:
    if self.server == "player":
      if action != 5:
        return
      lateral = int(np.sign(self.player[1] - 5))
      self.ball = self.player.copy()
      self.ball_velocity = [-1, lateral]
      self.phase = "rally"
      return
    self.serve_delay -= 1
    if self.serve_delay > 0:
      return
    lateral = -1 if self.player[1] >= 5 else 1
    self.ball = self.opponent.copy()
    self.ball_velocity = [1, lateral]
    self.phase = "rally"

  def _player_return(self) -> None:
    if self.ball_velocity[0] <= 0 or self.ball[0] < 6:
      return
    row_distance = abs(self.player[0] - self.ball[0])
    column_offset = self.ball[1] - self.player[1]
    if row_distance > 1 or abs(column_offset) > 1:
      return
    lateral = int(np.sign(column_offset))
    if self.player[0] <= 6 and lateral == 0:
      lateral = -1 if self.opponent[1] >= 5 else 1
    self.ball_velocity = [-1, lateral]

  def _move_opponent(self) -> None:
    if self.ball_velocity[0] < 0:
      target_row = int(np.clip(self.ball[0] - 2, 0, 4))
      target_column = int(np.clip(self.ball[1] + self.ball_velocity[1] * 2, 1, 8))
    else:
      target_row = 1
      target_column = 5
    if self.opponent[1] != target_column:
      self.opponent[1] += int(np.sign(target_column - self.opponent[1]))
    elif self.opponent[0] != target_row:
      self.opponent[0] += int(np.sign(target_row - self.opponent[0]))

  def _opponent_return(self) -> None:
    if self.ball_velocity[0] >= 0 or self.ball[0] > 3:
      return
    if abs(self.opponent[0] - self.ball[0]) > 1 or abs(self.opponent[1] - self.ball[1]) > 1:
      return
    lateral = -1 if self.player[1] >= 5 else 1
    if self.opponent[0] >= 3:
      lateral = int(np.clip(lateral * 2, -1, 1))
    self.ball_velocity = [1, lateral]

  def _move_ball(self) -> None:
    self.ball[0] += self.ball_velocity[0]
    self.ball[1] += self.ball_velocity[1]
    if self.ball[1] <= 0 or self.ball[1] >= 9:
      self.ball[1] = int(np.clip(self.ball[1], 0, 9))
      self.ball_velocity[1] *= -1
    if self.ball[0] < 0:
      self._award_point("player")
    elif self.ball[0] > 9:
      self._award_point("opponent")

  def _award_point(self, winner: str) -> None:
    if winner == "player":
      self.player_points += 1
    else:
      self.opponent_points += 1
    if max(self.player_points, self.opponent_points) >= 4 and abs(
      self.player_points - self.opponent_points
    ) >= 2:
      self._award_game(winner)
      return
    self._reset_point()

  def _award_game(self, winner: str) -> None:
    if winner == "player":
      self.player_games += 1
    else:
      self.opponent_games += 1
    self.player_points = 0
    self.opponent_points = 0
    games_played = self.player_games + self.opponent_games
    self.server = "player" if games_played % 2 == 0 else "opponent"
    self.side_swapped = games_played % 2 == 1
    if self.player_games == 6 and self.opponent_games == 6:
      self.player_games = 0
      self.opponent_games = 0
      self.tiebreak_reset = True
    lead = self.player_games - self.opponent_games
    if self.tiebreak_reset and abs(lead) >= 2:
      self.winner = "player" if lead > 0 else "opponent"
      self.terminal = True
    elif not self.tiebreak_reset and max(self.player_games, self.opponent_games) >= 6 and abs(lead) >= 2:
      self.winner = "player" if lead > 0 else "opponent"
      self.terminal = True
    if not self.terminal:
      self._reset_point()

  def _reset_point(self) -> None:
    self.phase = "serve"
    self.serve_delay = 2
    self.player = [8, 5]
    self.opponent = [1, 5]
    self.ball = (self.player if self.server == "player" else self.opponent).copy()
    self.ball_velocity = [0, 0]

  def _progress_score(self) -> float:
    if self.terminal and self.winner == "player":
      return self.score_cap
    return float(min(6, self.player_games))

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    state[*self.opponent, 1] = True
    state[*self.ball, 2] = True
    state[4:6, :, 3] = True
    state[*self.previous_player, 4] = True
    state[*self.previous_opponent, 4] = True
    state[*self.previous_ball, 4] = True
    values = (
      (self.player_points, 4),
      (self.opponent_points, 4),
      (self.player_games, 3),
      (self.opponent_games, 3),
      (self.ball_velocity[0] + 1, 2),
      (self.ball_velocity[1] + 1, 2),
      (self.steps, 11),
    )
    offset = 0
    for value, width in values:
      for bit in range(width):
        index = offset + bit
        state[index // 10, index % 10, 5] = bool(value & (1 << bit))
      offset += width
    state[3, 0, 5] = self.server == "player"
    state[3, 1, 5] = self.phase == "rally"
    state[3, 2, 5] = self.tiebreak_reset
    state[3, 3, 5] = self.side_swapped
    return state
