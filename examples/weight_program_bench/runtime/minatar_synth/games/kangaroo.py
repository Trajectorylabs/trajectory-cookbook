import numpy as np


FRUIT_VALUES = (100, 200, 400, 800)
FRUIT_POSITIONS = ((7, 2), (4, 5), (1, 7))


class KangarooEnvironment:
  """Three-screen rescue cycle with ladders, leaps, bells, and boxing."""

  score_cap = 74_700.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 1_800):
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
    self.round = 0
    self.screen = 0
    self.lives = 3
    self.score = 0.0
    self.next_bonus_life = 20_000.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed_monkeys: set[tuple[int, int, int]] = set()
    self.claimed_apples: set[tuple[int, int, int]] = set()
    self.claimed_fruit: set[tuple[int, int, int, int]] = set()
    self._start_screen()

  def _start_screen(self) -> None:
    self.player = [8, 0]
    self.previous_player = self.player.copy()
    self.baby = [1, 9]
    self.ladders = ({2, 6, 8}, {1, 5, 8}, {3, 6, 9})[self.screen]
    self.gaps = set()
    if self.screen == 1:
      self.gaps = {(8, 3), (8, 4), (5, 6), (5, 7)}
    elif self.screen == 2:
      self.gaps = {(8, 4), (5, 2), (5, 3)}
    monkey_positions = ((8, 8), (5, 3), (2, 8), (5, 8))
    self.monkeys = {
      index: list(position)
      for index, position in enumerate(monkey_positions)
      if (self.round, self.screen, index) not in self.claimed_monkeys
    }
    self.monkey_directions = {index: -1 for index in self.monkeys}
    self.apples: list[dict[str, object]] = []
    self.spawned_apples: set[int] = set()
    self.fruit_tiers = {index: 0 for index in range(3)}
    self.available_fruit = {
      index for index in range(3) if (self.round, self.screen, 0, index) not in self.claimed_fruit
    }
    self.collected_for_bell: set[int] = set()
    self.bell_rings = 0
    self.bell = [1, 5]
    self.timer = 2_000
    self.screen_steps = 0
    self.jump_timer = 0
    self.ducking = False
    self.previous_apples: list[list[int]] = []

  @property
  def fruit_tier(self) -> int:
    return max(self.fruit_tiers.values())

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    lives_before = self.lives
    screen_before = (self.round, self.screen)
    self.ducking = action == 4
    self._move_player(action)
    if self.lives != lives_before or self.terminal:
      return self._finish_step(0.0, consume_timer=False)
    reward = self._collect_fruit_and_ring_bell()
    if action == 5:
      reward += self._punch()
    self._spawn_apples()
    self._move_monkeys()
    self._move_apples()
    self._resolve_contacts(action)
    if self.lives != lives_before or self.terminal:
      return self._finish_step(reward, consume_timer=False)
    if self.player == self.baby:
      reward += self._rescue_baby()
    transitioned = (self.round, self.screen) != screen_before
    return self._finish_step(reward, consume_timer=not transitioned)

  def _finish_step(self, reward: float, consume_timer: bool = True) -> tuple[float, bool]:
    self.score += reward
    while self.score >= self.next_bonus_life:
      self.lives = min(20, self.lives + 1)
      self.next_bonus_life += 20_000.0
    if consume_timer:
      self.timer -= 20
      self.screen_steps += 1
      self.jump_timer = max(0, self.jump_timer - 1)
      if self.timer <= 0:
        self._lose_life()
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    if action in (1, 3):
      delta = -1 if action == 1 else 1
      distance = 2 if self.jump_timer > 0 else 1
      target_column = int(np.clip(self.player[1] + delta * distance, 0, 9))
      crossed = range(min(self.player[1], target_column), max(self.player[1], target_column) + 1)
      if self.player[0] in (5, 8) and any((self.player[0], column) in self.gaps for column in crossed):
        if self.jump_timer == 0:
          self._lose_life()
          return
      self.player[1] = target_column
    elif action == 2:
      if self.player[1] in self.ladders and self.player[0] > 1:
        self.player[0] -= 1
      else:
        self.jump_timer = 2
        self.player[0] = max(0, self.player[0] - 1)
    elif action == 4 and self.player[1] in self.ladders and self.player[0] < 8:
      self.player[0] += 1
    elif self.jump_timer == 0 and self.player[0] not in (1, 2, 5, 8):
      self.player[0] = min(8, self.player[0] + 1)

  def _collect_fruit_and_ring_bell(self) -> float:
    reward = 0.0
    for index in list(self.available_fruit):
      if self.player == list(FRUIT_POSITIONS[index]):
        tier = self.fruit_tiers[index]
        identity = (self.round, self.screen, tier, index)
        self.available_fruit.remove(index)
        self.collected_for_bell.add(index)
        if identity not in self.claimed_fruit:
          self.claimed_fruit.add(identity)
          reward += float(FRUIT_VALUES[tier])
    if self.player == self.bell and self.collected_for_bell and self.bell_rings < 3:
      self.bell_rings += 1
      for index in self.collected_for_bell:
        self.fruit_tiers[index] = min(3, self.fruit_tiers[index] + 1)
        identity = (self.round, self.screen, self.fruit_tiers[index], index)
        if identity not in self.claimed_fruit:
          self.available_fruit.add(index)
      self.collected_for_bell = set()
    return reward

  def _punch(self) -> float:
    reward = 0.0
    for index, monkey in list(self.monkeys.items()):
      if abs(monkey[0] - self.player[0]) + abs(monkey[1] - self.player[1]) <= 1:
        del self.monkeys[index]
        identity = (self.round, self.screen, index)
        if identity not in self.claimed_monkeys:
          self.claimed_monkeys.add(identity)
          reward += 200.0
        return reward
    for apple in list(self.apples):
      position = apple["position"]
      if abs(position[0] - self.player[0]) + abs(position[1] - self.player[1]) <= 1:
        self.apples.remove(apple)
        identity = (self.round, self.screen, int(apple["identity"]))
        if identity not in self.claimed_apples:
          self.claimed_apples.add(identity)
          reward += 200.0 if apple["kind"] == "falling" else 100.0
        return reward
    return reward

  def _spawn_apples(self) -> None:
    interval = max(5, 10 - self.round)
    if self.screen_steps % interval or len(self.spawned_apples) >= 8 or not self.monkeys:
      return
    identity = len(self.spawned_apples)
    self.spawned_apples.add(identity)
    monkey = self.monkeys[min(self.monkeys)]
    kinds = ("low", "high", "middle", "falling")
    kind = kinds[(identity + self.round) % len(kinds)]
    position = [max(0, monkey[0] - 2), monkey[1]] if kind == "falling" else monkey.copy()
    self.apples.append(
      {
        "identity": identity,
        "kind": kind,
        "position": position,
        "direction": int(np.sign(self.player[1] - monkey[1])) or -1,
      }
    )

  def _move_monkeys(self) -> None:
    if self.screen_steps % max(2, 4 - self.round // 2):
      return
    for index, monkey in self.monkeys.items():
      direction = int(np.sign(self.player[1] - monkey[1]))
      monkey[1] = int(np.clip(monkey[1] + direction, 0, 9))
      self.monkey_directions[index] = direction

  def _move_apples(self) -> None:
    self.previous_apples = [list(apple["position"]) for apple in self.apples]
    for apple in list(self.apples):
      if apple["kind"] == "falling":
        apple["position"][0] += 1
      else:
        apple["position"][1] += int(apple["direction"])
      position = apple["position"]
      if not (0 <= position[0] < 10 and 0 <= position[1] < 10):
        self.apples.remove(apple)

  def _resolve_contacts(self, action: int) -> None:
    if any(monkey == self.player for monkey in self.monkeys.values()):
      self._lose_life()
      return
    for apple in self.apples:
      if apple["position"] != self.player:
        continue
      avoided = (apple["kind"] == "low" and action == 2) or (apple["kind"] == "high" and self.ducking)
      if not avoided:
        self._lose_life()
      return

  def _rescue_baby(self) -> float:
    reward = float(self.timer)
    self.screen += 1
    if self.screen >= 3:
      self.screen = 0
      self.round += 1
    if self.round >= 3:
      self.terminal = True
    else:
      self._start_screen()
    return reward

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    else:
      self._start_screen()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    for monkey in self.monkeys.values():
      state[*monkey, 1] = True
      state[*monkey, 3] = True
    for apple in self.apples:
      state[*apple["position"], 1] = True
      if apple["kind"] in ("high", "middle"):
        state[*apple["position"], 2] = True
      if apple["kind"] == "middle":
        state[*apple["position"], 3] = True
      if apple["kind"] == "falling":
        state[*apple["position"], 4] = True
    state[*self.baby, 2] = True
    for index in self.available_fruit:
      state[*FRUIT_POSITIONS[index], 2] = True
    state[*self.bell, 2] = True
    for row in (2, 5, 8):
      for column in range(10):
        if (row, column) not in self.gaps:
          state[row, column, 3] = True
    for column in self.ladders:
      state[1:9, column, 3] = True
    state[*self.previous_player, 4] = True
    for apple in self.previous_apples:
      state[*apple, 4] = True
    current_claims = []
    current_claims.extend(
      (self.round, self.screen, index) in self.claimed_monkeys for index in range(4)
    )
    current_claims.extend(
      (self.round, self.screen, index) in self.claimed_apples for index in range(8)
    )
    current_claims.extend(
      (self.round, self.screen, tier, index) in self.claimed_fruit
      for tier in range(4)
      for index in range(3)
    )
    for index, claimed in enumerate(current_claims):
      state[index // 10, index % 10, 5] = claimed
    for index in self.spawned_apples:
      state[3, index, 5] = True
    for index, tier in self.fruit_tiers.items():
      state[4, index * 2, 5] = bool(tier & 1)
      state[4, index * 2 + 1, 5] = bool(tier & 2)
    state[9, self.round, 5] = True
    state[9, 3 + self.screen, 5] = True
    for bit in range(5):
      state[8, bit, 5] = bool(self.lives & (1 << bit))
    for bit in range(7):
      state[7, bit, 5] = bool((self.timer // 20) & (1 << bit))
    state[6, 4 + self.bell_rings, 5] = True
    state[5, : self.jump_timer, 5] = True
    state[5, 2, 5] = self.ducking
    return state
