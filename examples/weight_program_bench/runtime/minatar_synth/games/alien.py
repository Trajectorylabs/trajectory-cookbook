import numpy as np


MAZE = np.array([
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
  [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
  [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=bool)


class AlienEnvironment:
  """Maze pursuit with eggs, pulsars, and three vulnerable Alien enemies."""

  score_cap = 35.0

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
    self.player = [8, 5]
    self.previous_player = self.player.copy()
    self.aliens = [[1, 1], [1, 8], [7, 1]]
    self.previous_aliens = [position.copy() for position in self.aliens]
    passages = [tuple(position) for position in np.argwhere(~MAZE)]
    reserved = {tuple(self.player), *(tuple(position) for position in self.aliens)}
    self.pulsars = {(3, 3), (5, 8), (8, 1)}
    self.eggs = {position for position in passages if position not in reserved | self.pulsars}
    self.eggs = set(sorted(self.eggs)[:20])
    self.power_timer = 0
    self.last_action = 0
    self.steps = 0
    self.terminal = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    delta = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}.get(action, (0, 0))
    target = [self.player[0] + delta[0], self.player[1] + delta[1]]
    if not MAZE[*target]:
      self.player = target

    reward = 0.0
    position = tuple(self.player)
    if position in self.eggs:
      self.eggs.remove(position)
      reward += 1.0
    if position in self.pulsars:
      self.pulsars.remove(position)
      self.power_timer = 30

    self.previous_aliens = [alien.copy() for alien in self.aliens]
    for alien in self.aliens:
      self._move_alien(alien)
    defeated = []
    for alien in self.aliens:
      if alien == self.player:
        if self.power_timer > 0:
          reward += 5.0
          defeated.append(alien)
        else:
          self.terminal = True
    for alien in defeated:
      self.aliens.remove(alien)
    self.power_timer = max(0, self.power_timer - 1)
    self.steps += 1
    if not self.eggs:
      self.terminal = True
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_alien(self, alien: list[int]) -> None:
    candidates = []
    for row_delta, column_delta in ((0, -1), (-1, 0), (0, 1), (1, 0)):
      candidate = [alien[0] + row_delta, alien[1] + column_delta]
      if not MAZE[*candidate]:
        distance = abs(candidate[0] - self.player[0]) + abs(candidate[1] - self.player[1])
        candidates.append((distance, self.random.rand(), candidate))
    if candidates:
      alien[:] = min(candidates)[2]

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for alien in self.aliens:
      state[*alien, 1] = True
    for egg in self.eggs:
      state[*egg, 2] = True
    for pulsar in self.pulsars:
      state[*pulsar, 3] = True
    for previous in self.previous_aliens:
      state[*previous, 4] = True
    state[9, : min(10, (self.power_timer + 2) // 3), 5] = True
    return state
