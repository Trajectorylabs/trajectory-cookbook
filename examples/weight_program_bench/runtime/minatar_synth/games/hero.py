import numpy as np


class HeroEnvironment:
  """Twenty-level prop-pack mine rescue with finite reward identities."""

  score_cap = 79_700.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 6_000):
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
    self.level = 1
    self.lives = 4
    self.score = 0.0
    self.next_bonus_life = 20_000.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed_enemies: set[tuple[int, int]] = set()
    self.claimed_walls: set[tuple[int, int]] = set()
    self._start_level()

  def _start_level(self) -> None:
    self.player = [0, 1]
    self.previous_player = self.player.copy()
    self.facing = 1
    self.power = max(140, 300 - 5 * (self.level - 1))
    self.dynamite = 6
    self.laser: list[int] | None = None
    self.laser_direction = 1
    self.bomb: dict[str, object] | None = None
    wall_columns = (3, 6)
    wall_rows = (3 + self.level % 2, 6 - self.level % 2)
    self.walls = {
      (row, column): 3
      for row, column in zip(wall_rows, wall_columns, strict=True)
      if (self.level, row * 10 + column) not in self.claimed_walls
    }
    enemy_positions = ((2, 7), (5, 2), (7, 7))
    self.enemies = {
      index: [row, (column + self.level) % 9]
      for index, (row, column) in enumerate(enemy_positions)
      if (self.level, index) not in self.claimed_enemies
    }
    self.enemy_directions = {index: 1 if (index + self.level) % 2 else -1 for index in self.enemies}
    self.miner = [8, 8]
    self.lava_columns = set(range(2, 8)) if self.level >= 5 else set()
    self.raft = [8, 2 + self.level % 6]
    self.raft_direction = 1
    self.lantern = [1, 8]
    self.lantern_lit = True
    self.level_steps = 0

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    level_before = self.level
    lives_before = self.lives
    reward = self._apply_action(action)
    reward += self._move_laser()
    reward += self._tick_bomb()
    if self.level != level_before or self.lives != lives_before:
      return self._finish_step(reward, consume_power=False)
    if self.terminal:
      return self._finish_step(reward)
    self._apply_gravity(action)
    if self.terminal:
      return self._finish_step(reward)
    self._move_enemies()
    self._move_raft()
    self._resolve_contacts()
    if not self.terminal and self.player == self.miner:
      reward += self._rescue_miner()
    transitioned = self.level != level_before or self.lives != lives_before
    return self._finish_step(reward, consume_power=not transitioned)

  def _finish_step(self, reward: float, consume_power: bool = True) -> tuple[float, bool]:
    self.score += reward
    while self.score >= self.next_bonus_life:
      self.lives = min(20, self.lives + 1)
      self.next_bonus_life += 20_000.0
    if consume_power:
      self.power -= 1
      self.level_steps += 1
    self.steps += 1
    if self.power <= 0 and not self.terminal:
      self._lose_life()
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _apply_action(self, action: int) -> float:
    if action == 1:
      self.player[1] = max(0, self.player[1] - 1)
      self.facing = -1
    elif action == 2:
      self.player[0] = max(0, self.player[0] - 1)
    elif action == 3:
      self.player[1] = min(9, self.player[1] + 1)
      self.facing = 1
    elif action == 4 and self.dynamite > 0 and self.bomb is None:
      self.dynamite -= 1
      self.bomb = {"position": self.player.copy(), "fuse": 3}
    elif action == 5 and self.laser is None:
      self.laser = self.player.copy()
      self.laser_direction = self.facing
    return 0.0

  def _move_laser(self) -> float:
    if self.laser is None:
      return 0.0
    self.laser[1] += self.laser_direction
    position = tuple(self.laser)
    for index, enemy in list(self.enemies.items()):
      if enemy == self.laser:
        del self.enemies[index]
        self.claimed_enemies.add((self.level, index))
        self.laser = None
        return 50.0
    if position in self.walls:
      self.walls[position] -= 1
      if self.walls[position] <= 0:
        del self.walls[position]
        self.claimed_walls.add((self.level, position[0] * 10 + position[1]))
      self.laser = None
    elif self.laser == self.lantern:
      self.lantern_lit = False
      self.laser = None
    elif not 0 <= self.laser[1] < 10:
      self.laser = None
    return 0.0

  def _tick_bomb(self) -> float:
    if self.bomb is None:
      return 0.0
    self.bomb["fuse"] = int(self.bomb["fuse"]) - 1
    if self.bomb["fuse"]:
      return 0.0
    bomb_position = list(self.bomb["position"])
    self.bomb = None
    reward = 0.0
    for position in list(self.walls):
      if abs(position[0] - bomb_position[0]) + abs(position[1] - bomb_position[1]) <= 2:
        del self.walls[position]
        identity = (self.level, position[0] * 10 + position[1])
        if identity not in self.claimed_walls:
          self.claimed_walls.add(identity)
          reward += 75.0
    for index, enemy in list(self.enemies.items()):
      if abs(enemy[0] - bomb_position[0]) + abs(enemy[1] - bomb_position[1]) <= 2:
        del self.enemies[index]
        identity = (self.level, index)
        if identity not in self.claimed_enemies:
          self.claimed_enemies.add(identity)
          reward += 50.0
    if abs(self.player[0] - bomb_position[0]) + abs(self.player[1] - bomb_position[1]) <= 1:
      self._lose_life()
    return reward

  def _apply_gravity(self, action: int) -> None:
    if action != 2:
      target = [min(8, self.player[0] + 1), self.player[1]]
      if tuple(target) not in self.walls:
        self.player = target
    if tuple(self.player) in self.walls:
      self.player = self.previous_player.copy()

  def _move_enemies(self) -> None:
    if self.level_steps % 2:
      return
    for index, enemy in self.enemies.items():
      next_column = enemy[1] + self.enemy_directions[index]
      if next_column < 0 or next_column > 9 or (enemy[0], next_column) in self.walls:
        self.enemy_directions[index] *= -1
      else:
        enemy[1] = next_column

  def _move_raft(self) -> None:
    if not self.lava_columns:
      return
    carried = self.player == self.raft
    self.raft[1] += self.raft_direction
    if self.raft[1] <= min(self.lava_columns) or self.raft[1] >= max(self.lava_columns):
      self.raft_direction *= -1
    if carried:
      self.player = self.raft.copy()

  def _resolve_contacts(self) -> None:
    if any(enemy == self.player for enemy in self.enemies.values()):
      self._lose_life()
      return
    if self.player[0] == 8 and self.player[1] in self.lava_columns and self.player != self.raft:
      self._lose_life()

  def _rescue_miner(self) -> float:
    reward = 1_000.0 + 50.0 * self.dynamite + float(self.power * self.level)
    self.level += 1
    if self.level > 20:
      self.terminal = True
    else:
      self._start_level()
    return reward

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    else:
      self._start_level()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for enemy in self.enemies.values():
      state[*enemy, 1] = True
    for column in self.lava_columns:
      state[8, column, 1] = True
    state[*self.miner, 2] = True
    state[*self.raft, 2] = True
    if self.lantern_lit:
      state[*self.lantern, 2] = True
    for wall in self.walls:
      state[*wall, 3] = True
    if self.laser is not None:
      state[*self.laser, 3] = True
    if self.bomb is not None:
      state[*self.bomb["position"], 3] = True
    state[*self.previous_player, 4] = True
    for bit in range(5):
      state[9, bit, 5] = bool(self.level & (1 << bit))
    for bit in range(3):
      state[7, bit, 5] = bool(self.dynamite & (1 << bit))
    for bit in range(9):
      state[6, bit, 5] = bool(self.power & (1 << bit))
    for bit in range(5):
      state[5, bit, 5] = bool(self.lives & (1 << bit))
    state[4, 0, 5] = self.facing > 0
    state[4, 1, 5] = self.lantern_lit
    if self.bomb is not None:
      state[4, 2 + int(self.bomb["fuse"]), 5] = True
    return state
