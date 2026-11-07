import numpy as np


ITEMS = ("gun", "button", "comb", "sole", "money", "vase", "necklace", "stamp")
ITEM_BLOCKS = {item: 2 + index * 4 for index, item in enumerate(ITEMS)}
DESTINATION_BLOCKS = {item: 3 + index * 4 for index, item in enumerate(ITEMS)}
EMPTY_CHARACTER_BLOCKS = (1, 5, 9, 13, 17, 21, 25, 29)
SECRET_PASSAGES = {0: 12, 7: 20, 14: 28, 21: 3, 28: 16}
DEMERITS = {"brick": 100, "flowerpot": 200, "rat": 300, "bird": 400, "dagger": 1_000}


class PrivateEyeEnvironment:
  """Case Five city search with one-item evidence logistics."""

  score_cap = 251_800.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 800):
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
    self.block = 0
    self.car = [7, 1]
    self.previous_car = self.car.copy()
    self.facing = 1
    self.score = 1_000.0
    self.steps = 0
    self.last_action = 0
    self.jump_timer = 0
    self.terminal = False
    self.carried: str | None = None
    self.acquired: set[str] = set()
    self.returned: set[str] = set()
    self.nabbed_empty: set[int] = set()
    self.item_blocks = dict(ITEM_BLOCKS)
    self.le_fiend_nabbed = False
    self.hazard = [1, 8]
    self.previous_hazard = self.hazard.copy()
    self.hazard_kind = "brick"
    self.hazard_direction = -1

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_car = self.car.copy()
    self.previous_hazard = self.hazard.copy()
    block_before = self.block
    reward = self._move_car(action)
    reward += self._interact(action)
    if self.block == block_before and not self.terminal:
      self._move_hazard()
      reward += self._resolve_hazards()
    reward += self._deliver_item()
    self.score = max(0.0, self.score + reward)
    self.steps += 1
    self.jump_timer = max(0, self.jump_timer - 1)
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_car(self, action: int) -> float:
    if action == 1:
      self.car[1] -= 1
      self.facing = -1
    elif action == 3:
      self.car[1] += 1
      self.facing = 1
    elif action == 5:
      self.jump_timer = 3
      self.car[0] = 4
    elif action == 2 and self.block in SECRET_PASSAGES and self.car[1] == 5:
      self.block = SECRET_PASSAGES[self.block]
      self.car = [7, 5]
      self.previous_car = self.car.copy()
      self._reset_hazard()
      return 0.0
    if self.jump_timer == 0:
      self.car[0] = 7
    if self.car[1] < 0:
      self._change_block(-1)
    elif self.car[1] > 9:
      self._change_block(1)
    return 0.0

  def _change_block(self, direction: int) -> None:
    self.block = (self.block + direction) % 32
    self.car = [7, 9 if direction < 0 else 0]
    self.previous_car = self.car.copy()
    self._reset_hazard()

  def _reset_hazard(self) -> None:
    kinds = ("brick", "flowerpot", "rat", "bird", "pothole")
    if self.returned:
      kinds = (*kinds, "dagger")
    self.hazard_kind = kinds[self.block % len(kinds)]
    self.hazard = [1 if self.hazard_kind in ("brick", "flowerpot", "bird", "dagger") else 7, 8]
    self.previous_hazard = self.hazard.copy()
    self.hazard_direction = -1

  def _interact(self, action: int) -> float:
    if action != 5 or self.car[1] != 5:
      return 0.0
    for item, block in self.item_blocks.items():
      if block != self.block or item in self.returned:
        continue
      previous_item = self.carried
      self.carried = item
      self.item_blocks[item] = -1
      if previous_item in ITEMS:
        self.item_blocks[previous_item] = self.block
      if item in self.acquired:
        return 0.0
      self.acquired.add(item)
      return 5_000.0 if ITEMS.index(item) < 4 else 15_000.0
    if self.block in EMPTY_CHARACTER_BLOCKS and self.block not in self.nabbed_empty:
      self.nabbed_empty.add(self.block)
      return 100.0
    if self.block == 16 and len(self.returned) == 8 and not self.le_fiend_nabbed:
      self.le_fiend_nabbed = True
      self.carried = "fiend"
      return 25_000.0
    return 0.0

  def _deliver_item(self) -> float:
    if self.carried in ITEMS and self.block == DESTINATION_BLOCKS[self.carried] and self.car[1] == 8:
      item = self.carried
      self.returned.add(item)
      self.carried = None
      return 10_000.0 if ITEMS.index(item) < 4 else 20_000.0
    if self.carried == "fiend" and self.block == 0 and self.car[1] == 8:
      self.carried = None
      self.terminal = True
      return 25_000.0
    return 0.0

  def _move_hazard(self) -> None:
    if self.hazard_kind == "pothole":
      return
    if self.hazard_kind in ("brick", "flowerpot", "bird", "dagger"):
      self.hazard[0] += 1
      if self.hazard[0] > 7:
        self.hazard = [1, int(self.random.randint(1, 9))]
    else:
      self.hazard[1] += self.hazard_direction
      if self.hazard[1] in (1, 8):
        self.hazard_direction *= -1

  def _resolve_hazards(self) -> float:
    if self.jump_timer:
      return 0.0
    if self.block % 6 == 0 and self.car[1] == 6:
      return -1.0
    if self.block % 8 == 4 and self.car[1] == 5:
      return -100.0
    if self.hazard == self.car:
      penalty = DEMERITS.get(self.hazard_kind, 1)
      if self.hazard_kind == "dagger" and self.carried is not None:
        if self.carried in ITEMS:
          self.item_blocks[self.carried] = (self.block - self.facing) % 32
        self.carried = None
      return -float(penalty)
    return 0.0

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.car, 0] = True
    state[*self.previous_car, 4] = True
    state[*self.hazard, 1] = True
    state[*self.previous_hazard, 4] = True
    for item, block in self.item_blocks.items():
      if block == self.block and item not in self.returned:
        state[4, 5, 2] = True
        state[4, ITEMS.index(item), 3] = True
    if self.block in EMPTY_CHARACTER_BLOCKS and self.block not in self.nabbed_empty:
      state[4, 5, 2] = True
    if self.block == 16 and len(self.returned) == 8 and not self.le_fiend_nabbed:
      state[4, 5, 2] = True
    if self.block in SECRET_PASSAGES:
      state[7, 5, 3] = True
    if self.block % 8 == 4:
      state[7, 5, 1] = True
    for index, item in enumerate(ITEMS):
      state[index // 10, index % 10, 5] = item in self.acquired
      state[1 + index // 10, index % 10, 5] = item in self.returned
    for index, block in enumerate(EMPTY_CHARACTER_BLOCKS):
      state[2, index, 5] = block in self.nabbed_empty
    carried_codes = {None: 0, **{item: index + 1 for index, item in enumerate(ITEMS)}, "fiend": 9}
    code = carried_codes[self.carried]
    for bit in range(4):
      state[3, bit, 5] = bool(code & (1 << bit))
    for bit in range(5):
      state[3, 4 + bit, 5] = bool(self.block & (1 << bit))
    for bit in range(10):
      state[4, bit, 5] = bool(self.steps & (1 << bit))
    for item_index, item in enumerate(ITEMS):
      block_code = 32 if self.item_blocks[item] < 0 else self.item_blocks[item]
      for bit in range(6):
        index = 52 + item_index * 6 + bit
        state[index // 10, index % 10, 5] = bool(block_code & (1 << bit))
    facing_column = min(9, max(0, self.car[1] + self.facing))
    state[self.car[0], facing_column, 3] = True
    state[9, : self.jump_timer, 3] = True
    return state
