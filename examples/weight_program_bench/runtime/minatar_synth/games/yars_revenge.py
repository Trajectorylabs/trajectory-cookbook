import numpy as np


DIRECTIONS = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}
NEUTRAL_COLUMNS = frozenset((4, 5))


class YarsRevengeEnvironment:
  """Four-Qotile normal-game campaign with shield, neutral zone, and Zorlon cannon."""

  score_cap = 40_224.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 4_000):
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
    self.stage = 0
    self.lives = 4
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.facing = 3
    self.terminal = False
    self._start_stage()

  def _start_stage(self) -> None:
    self.yar = [5, 2]
    self.qotile = [5, 9]
    self.qotile_state = "base"
    self.swirl_timer = 0
    self.shield = {(row, column) for row in range(1, 9) for column in range(6, 9)}
    self.cannon_ready = False
    self.energy_missile: list[int] | None = None
    self.zorlon: list[int] | None = None
    self.destroyer = [1, 1]

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    reward = 0.0
    if action in DIRECTIONS:
      self.facing = action
      reward += self._move_yar(action)
    elif action == 5:
      self._fire()
    self._move_energy_missile()
    reward += self._resolve_energy_hit()
    self._move_zorlon()
    reward += self._resolve_zorlon_hit()
    self._move_destroyer()
    self._move_swirl()
    self._resolve_hazards()
    self.score += reward
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_yar(self, action: int) -> float:
    row_delta, column_delta = DIRECTIONS[action]
    destination = [(self.yar[0] + row_delta) % 10, self.yar[1] + column_delta]
    if not 0 <= destination[1] < 10:
      return 0.0
    self.yar = destination
    position = tuple(self.yar)
    if position in self.shield:
      self.shield.remove(position)
      self.cannon_ready = True
      return 169.0
    if self.yar == self.qotile:
      self.cannon_ready = True
    return 0.0

  def _fire(self) -> None:
    if self.yar[1] in NEUTRAL_COLUMNS:
      return
    if self.cannon_ready and self.zorlon is None:
      self.zorlon = [self.yar[0], 0, 0, 1]
      self.cannon_ready = False
      return
    if self.energy_missile is None:
      row_delta, column_delta = DIRECTIONS[self.facing]
      self.energy_missile = [self.yar[0], self.yar[1], row_delta, column_delta]

  def _move_energy_missile(self) -> None:
    if self.energy_missile is None:
      return
    self.energy_missile[0] = (self.energy_missile[0] + self.energy_missile[2]) % 10
    self.energy_missile[1] += self.energy_missile[3]
    if not 0 <= self.energy_missile[1] < 10:
      self.energy_missile = None

  def _resolve_energy_hit(self) -> float:
    if self.energy_missile is None or tuple(self.energy_missile[:2]) not in self.shield:
      return 0.0
    self.shield.remove(tuple(self.energy_missile[:2]))
    self.energy_missile = None
    return 69.0

  def _move_zorlon(self) -> None:
    if self.zorlon is None:
      return
    self.zorlon[1] += 1
    if self.zorlon[1] >= 10:
      self.zorlon = None

  def _resolve_zorlon_hit(self) -> float:
    if self.zorlon is None:
      return 0.0
    position = tuple(self.zorlon[:2])
    if position in self.shield:
      self.shield.remove(position)
      self.zorlon = None
      return 69.0
    if self.zorlon[:2] != self.qotile:
      return 0.0
    reward = 6_000.0 if self.qotile_state == "flying" else 2_000.0 if self.qotile_state == "windup" else 1_000.0
    if reward == 6_000:
      self.lives += 1
    self.zorlon = None
    self.stage += 1
    if self.stage == 4:
      self.terminal = True
    else:
      self._start_stage()
    return reward

  def _move_destroyer(self) -> None:
    interval = max(1, 4 - self.stage)
    if self.steps % interval:
      return
    self.destroyer[0] += int(np.sign(self.yar[0] - self.destroyer[0]))
    self.destroyer[1] += int(np.sign(self.yar[1] - self.destroyer[1]))

  def _move_swirl(self) -> None:
    phase = self.steps % max(20, 40 - 5 * self.stage)
    if phase == 12:
      self.qotile_state = "windup"
      self.swirl_timer = 4
    elif self.qotile_state == "windup":
      self.swirl_timer -= 1
      if self.swirl_timer == 0:
        self.qotile_state = "flying"
    elif self.qotile_state == "flying":
      self.qotile[0] += int(np.sign(self.yar[0] - self.qotile[0]))
      self.qotile[1] += int(np.sign(self.yar[1] - self.qotile[1]))

  def _resolve_hazards(self) -> None:
    destroyer_hit = self.destroyer == self.yar and self.yar[1] not in NEUTRAL_COLUMNS
    swirl_hit = self.qotile_state == "flying" and self.qotile == self.yar
    own_cannon_hit = self.zorlon is not None and self.zorlon[:2] == self.yar
    if not (destroyer_hit or swirl_hit or own_cannon_hit):
      return
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    else:
      self.yar = [5, 2]
      self.destroyer = [1, 1]
      self.energy_missile = None
      self.zorlon = None

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[self.yar[0], self.yar[1], 0] = True
    state[self.destroyer[0], self.destroyer[1], 1] = True
    state[self.qotile[0], self.qotile[1], 1] = True
    for row, column in self.shield:
      state[row, column, 2] = True
    if self.energy_missile is not None:
      state[self.energy_missile[0], self.energy_missile[1], 3] = True
    if self.zorlon is not None:
      state[self.zorlon[0], self.zorlon[1], 3] = True
    for row in range(10):
      for column in NEUTRAL_COLUMNS:
        state[row, column, 4] = True
    values = ((self.stage, 3), (self.lives, 4), (len(self.shield), 5), (self.swirl_timer, 3), (self.steps, 12))
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[3, 0, 5] = self.cannon_ready
    state[3, 1, 5] = self.qotile_state == "windup"
    state[3, 2, 5] = self.qotile_state == "flying"
    return state
