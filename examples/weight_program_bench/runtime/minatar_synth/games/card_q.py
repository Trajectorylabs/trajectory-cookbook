import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
COLUMNS = 5


class CardQEnvironment:
  score_cap = 250

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.player = self.random.integers(1, 14, (5, 5))
    self.player[0] = [1, 3, 3, 11, 13]
    self.enemy = self.random.integers(1, 14, (5, 5))
    self.piles = [2, 12]
    self.steps = self.last_action = self.player_pile = self.enemy_pile = 0
    self.shuffle_timer = self.shuffle_count = 0
    self.enemy_timer = 120.0
    self.enemy_column: int | None = None
    self.center = self.target = 40.0
    self.multiplier = 1
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _match(self, rank: int, preferred: int) -> int | None:
    for index in (preferred, 1 - preferred):
      if (rank - self.piles[index]) % 13 in (1, 12):
        return index
    return None

  def _place(self, cards: np.ndarray, column: int, pile: int) -> None:
    self.piles[pile] = int(cards[0, column])
    cards[:-1, column] = cards[1:, column]
    cards[-1, column] = self.random.integers(1, 14)
    self.shuffle_timer = self.shuffle_count = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    column = action - 1 if 1 <= action <= COLUMNS and action != self.last_action else None
    self.last_action = action
    self.steps += 1
    reward = 0
    for frame in range(FRAMES_PER_STEP):
      reward += self._advance_frame(column if frame == 0 else None)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self, column: int | None) -> int:
    self.shuffle_timer += 1
    if self.shuffle_timer > 60:
      player_can_play = any(self._match(rank, self.player_pile) is not None for rank in self.player[0])
      enemy_can_play = any(self._match(rank, self.enemy_pile) is not None for rank in self.enemy[0])
      if not player_can_play:
        self.enemy_timer *= 0.3
      self.shuffle_count += 1
      if not (player_can_play or enemy_can_play) or self.shuffle_count > 2:
        self.piles = self.random.integers(1, 14, 2).tolist()
        self.shuffle_count = 0
      self.shuffle_timer = 0

    reward = 0
    if column is not None:
      pile = self._match(self.player[0, column], self.player_pile)
      if pile is None:
        self.target += 5
        self.multiplier = 1
        self.shuffle_timer = self.shuffle_count = 0
      else:
        self._place(self.player, column, pile)
        self.player_pile = pile
        self.target -= 5
        reward = self.multiplier
        self.multiplier += 1

    self.enemy_timer -= 1
    if self.enemy_timer < 0:
      self._enemy_turn()
    self.center += (self.target - self.center) * 0.1
    if self.target < 16:
      self.target += (16 - self.target) * 0.1
    self.terminal = self.center > 94
    return reward

  def _enemy_turn(self) -> None:
    self.enemy_timer = self.random.uniform(50, 70)
    if self.enemy_column is not None:
      pile = self._match(self.enemy[0, self.enemy_column], self.enemy_pile)
      if pile is None:
        self.enemy_timer *= 3
      else:
        self._place(self.enemy, self.enemy_column, pile)
        self.enemy_pile = pile
        self.target += 5
        self.multiplier = 1

    self.enemy_column = None
    start = int(self.random.integers(COLUMNS))
    for offset in range(COLUMNS):
      column = (start + offset) % COLUMNS
      pile = self._match(self.enemy[0, column], self.enemy_pile)
      if pile is not None:
        self.enemy_column = column
        if pile != self.enemy_pile:
          self.enemy_timer *= 1.5
        break

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    cards = np.concatenate((self.enemy[::-1], self.player))
    for bit in range(4):
      state[:, :5, bit] = (cards & (1 << bit)) != 0
    state[4:6, :5, 4] = True
    if self.enemy_column is not None:
      state[4, self.enemy_column, 5] = True
    for index, rank in enumerate(self.piles):
      for bit in range(4):
        state[4 + index, 5, bit] = bool(rank & (1 << bit))
      state[4 + index, 5, 4:6] = index == self.player_pile, index == self.enemy_pile
    for row, value in enumerate((self.steps, self.multiplier, round(self.center), round(self.target),
                                 round(self.enemy_timer), self.shuffle_timer, self.shuffle_count, self.last_action)):
      for bit in range(10):
        state[row, 6 + bit // 5, bit % 5] = bool(value & (1 << bit))
    return state
