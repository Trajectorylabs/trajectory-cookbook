import numpy as np


class KrullEnvironment:
  """Two-quest marriage, desert, web, path, and fortress campaign."""

  score_cap = 9_360.0

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 2_400):
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
    self.quest = 0
    self.lives = 3
    self.glaives = 0
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.terminal = False
    self.claimed: set[tuple[int, str, int]] = set()
    self.prison_chunks: set[int] = set()
    self._start_marriage()

  def _base_stage(self, stage: str) -> None:
    self.stage = stage
    self.stage_steps = 0
    self.player = [8, 5]
    self.previous_player = self.player.copy()
    self.projectile: dict[str, object] | None = None
    self.enemy_projectiles: list[list[int]] = []

  def _start_marriage(self) -> None:
    self._base_stage("marriage")
    self.lyssa = [1, 5]
    self.slayers = {
      index: [1 + index % 4, 0 if index % 2 == 0 else 9]
      for index in range(8)
      if (self.quest, "slayer", index) not in self.claimed
    }

  def _start_desert(self) -> None:
    self._base_stage("desert")
    self.player = [5, 5]
    self.previous_player = self.player.copy()
    self.desert_items = [
      {"kind": "glaive", "index": index, "position": [7, 9 + index * 12]}
      for index in range(3)
      if (self.quest, "glaive", index) not in self.claimed
    ]
    if (self.quest, "life", 0) not in self.claimed:
      self.desert_items.append({"kind": "life", "index": 0, "position": [7, 27]})

  def _start_web(self) -> None:
    self._base_stage("web")
    self.player = [9, 5]
    self.previous_player = self.player.copy()
    self.cocoon = [0, 5]
    self.spider = [0, 0]
    self.web_timer = 80 - 10 * self.quest
    self.strands = {
      index: [row, (index * 3 + self.quest) % 10]
      for index, row in enumerate((8, 7, 6, 5, 4, 3, 2, 1))
    }

  def _start_path(self) -> None:
    self._base_stage("path")
    self.player = [5, 4]
    self.previous_player = self.player.copy()
    self.path_selection = 0
    self.sunrise = 0
    self.sun_timer = 30 - 5 * self.quest
    self.correct_path = (self.quest * 3 + self.sunrise * 5) % 8

  def _start_fortress(self) -> None:
    self._base_stage("fortress")
    self.beast = [3, 5]
    self.lyssa = [1, 5]
    self.lyssa_free = len(self.prison_chunks) >= 6
    self.has_fireball = False

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.previous_player = self.player.copy()
    stage_before = self.stage
    lives_before = self.lives
    reward = {
      "marriage": self._act_marriage,
      "desert": self._act_desert,
      "web": self._act_web,
      "path": self._act_path,
      "fortress": self._act_fortress,
    }[self.stage](action)
    transitioned = self.stage != stage_before or self.lives != lives_before
    self.score += reward
    self.steps += 1
    if not transitioned:
      self.stage_steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _move_player(self, action: int) -> None:
    if action in (1, 2, 3, 4):
      delta = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}[action]
      self.player[0] = int(np.clip(self.player[0] + delta[0], 0, 9))
      self.player[1] = int(np.clip(self.player[1] + delta[1], 0, 9))

  def _act_marriage(self, action: int) -> float:
    self._move_player(action)
    reward = 0.0
    if action == 5:
      for index, slayer in list(self.slayers.items()):
        if abs(slayer[0] - self.player[0]) + abs(slayer[1] - self.player[1]) <= 1:
          del self.slayers[index]
          identity = (self.quest, "slayer", index)
          if identity not in self.claimed:
            self.claimed.add(identity)
            reward = float(10 * (index + 1 + self.quest))
          break
    interval = max(1, 3 - self.quest)
    if self.stage_steps % interval == 0:
      for slayer in self.slayers.values():
        slayer[0] += int(np.sign(self.lyssa[0] - slayer[0]))
        slayer[1] += int(np.sign(self.lyssa[1] - slayer[1]))
    if any(slayer == self.player for slayer in self.slayers.values()):
      self._lose_life()
    elif self.stage_steps >= 79 or not self.slayers:
      self._start_desert()
    return reward

  def _act_desert(self, action: int) -> float:
    self._move_player(action)
    for item in self.desert_items:
      item["position"][1] -= 1
    if action == 5:
      for item in list(self.desert_items):
        if item["position"][1] == self.player[1]:
          identity = (self.quest, str(item["kind"]), int(item["index"]))
          if identity not in self.claimed:
            self.claimed.add(identity)
            if item["kind"] == "glaive":
              self.glaives = min(5, self.glaives + 1)
            else:
              self.lives = min(10, self.lives + 1)
          self.desert_items.remove(item)
          break
    self.desert_items = [item for item in self.desert_items if item["position"][1] >= 0]
    if self.stage_steps >= 59:
      self._start_web()
    return 0.0

  def _act_web(self, action: int) -> float:
    reward = 0.0
    if action == 5:
      crossed = [index for index, strand in self.strands.items() if strand[0] == self.player[0] - 1]
      if crossed:
        index = crossed[0]
        self.player[0] = max(0, self.player[0] - 2)
        identity = (self.quest, "strand", index)
        if identity not in self.claimed:
          self.claimed.add(identity)
          reward += float(40 + 10 * self.quest)
    else:
      self._move_player(action)
    self.spider[0] += int(np.sign(self.player[0] - self.spider[0]))
    self.spider[1] += int(np.sign(self.player[1] - self.spider[1]))
    self.web_timer -= 1
    if self.player == self.spider or self.web_timer <= 0:
      self._lose_life()
    elif self.player == self.cocoon:
      identity = (self.quest, "cocoon", 0)
      if identity not in self.claimed:
        self.claimed.add(identity)
        reward += 500.0
      self._start_path()
    return reward

  def _act_path(self, action: int) -> float:
    if action == 1:
      self.path_selection = (self.path_selection - 1) % 8
    elif action == 3:
      self.path_selection = (self.path_selection + 1) % 8
    elif action == 5:
      if self.path_selection == self.correct_path:
        self._start_fortress()
      else:
        if self.lives > 1:
          self.lives -= 1
        elif self.glaives > 0:
          self.glaives -= 1
        else:
          self._lose_life()
        if not self.terminal:
          self._start_web()
      return 0.0
    self.sun_timer -= 1
    if self.sun_timer <= 0:
      self.sunrise += 1
      self.correct_path = (self.quest * 3 + self.sunrise * 5) % 8
      self._start_web()
    return 0.0

  def _act_fortress(self, action: int) -> float:
    reward = 0.0
    self._move_player(action)
    if self.projectile is None and action == 5:
      if self.has_fireball:
        self.projectile = {"kind": "fireball", "position": self.player.copy(), "returning": False}
      elif self.glaives > 0:
        self.projectile = {"kind": "glaive", "position": self.player.copy(), "returning": False}
    if self.projectile is not None:
      reward += self._move_fortress_projectile(action)
    if self.stage != "fortress":
      return reward
    if self.lyssa_free and not self.has_fireball and self.lyssa[0] < self.player[0]:
      self.lyssa[0] += 1
      if self.lyssa[0] == self.player[0]:
        self.has_fireball = True
    if self.stage_steps % max(4, 7 - self.quest * 2) == 0:
      self.enemy_projectiles.append(self.beast.copy())
    for fireball in list(self.enemy_projectiles):
      fireball[0] += int(np.sign(self.player[0] - fireball[0]))
      fireball[1] += int(np.sign(self.player[1] - fireball[1]))
      if fireball == self.player:
        self._lose_life()
        return reward
    if self.glaives == 0 and self.projectile is None and self.player[1] in (0, 9):
      self._start_desert()
    return reward

  def _move_fortress_projectile(self, action: int) -> float:
    projectile = self.projectile
    position = projectile["position"]
    if projectile["kind"] == "fireball":
      position[0] -= 1
      position[1] += int(np.sign(self.beast[1] - position[1]))
      if position == self.beast:
        self.projectile = None
        return self._destroy_beast()
    elif not projectile["returning"]:
      position[0] -= 1
      if action == 1:
        position[1] = max(0, position[1] - 1)
      elif action == 3:
        position[1] = min(9, position[1] + 1)
      if position[0] <= 1 and 2 <= position[1] <= 7:
        chunk = position[1] - 2
        identity = (self.quest, "chunk", chunk)
        reward = 0.0
        if identity not in self.claimed:
          self.claimed.add(identity)
          self.prison_chunks.add(chunk)
          reward = 70.0
        projectile["returning"] = True
        if len(self.prison_chunks) >= 6:
          self.lyssa_free = True
        return reward
      if position == self.beast:
        self.glaives -= 1
        self.projectile = None
    else:
      position[0] += 1
      if position[0] >= self.player[0]:
        if position[1] != self.player[1]:
          self.glaives -= 1
        self.projectile = None
    if self.projectile is not None and not (0 <= position[0] < 10 and 0 <= position[1] < 10):
      if projectile["kind"] == "glaive":
        self.glaives -= 1
      self.projectile = None
    return 0.0

  def _destroy_beast(self) -> float:
    identity = (self.quest, "beast", 0)
    reward = 0.0 if identity in self.claimed else 3_000.0
    self.claimed.add(identity)
    self.quest += 1
    if self.quest >= 2:
      self.terminal = True
    else:
      self.prison_chunks = set()
      self._start_marriage()
    return reward

  def _lose_life(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
    elif self.stage == "marriage":
      self._start_marriage()
    elif self.stage in ("web", "path"):
      self._start_web()
    else:
      self._start_fortress()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[*self.player, 0] = True
    if self.stage == "marriage":
      for slayer in self.slayers.values():
        state[*slayer, 1] = True
      state[*self.lyssa, 2] = True
    elif self.stage == "desert":
      for item in self.desert_items:
        column = int(item["position"][1])
        if 0 <= column < 10:
          state[item["position"][0], column, 2] = True
    elif self.stage == "web":
      state[*self.spider, 1] = True
      state[*self.cocoon, 2] = True
      for strand in self.strands.values():
        state[*strand, 3] = True
    elif self.stage == "fortress":
      state[*self.beast, 1] = True
      state[*self.lyssa, 2] = True
      for chunk in range(6):
        if chunk not in self.prison_chunks:
          state[1, chunk + 2, 3] = True
      for fireball in self.enemy_projectiles:
        state[*fireball, 1] = True
    if self.projectile is not None:
      state[*self.projectile["position"], 3] = True
    state[*self.previous_player, 4] = True
    stages = ("marriage", "desert", "web", "path", "fortress")
    for index in range(8):
      state[0, index, 5] = (self.quest, "strand", index) in self.claimed
    state[0, 8, 5] = (self.quest, "cocoon", 0) in self.claimed
    state[0, 9, 5] = (self.quest, "beast", 0) in self.claimed
    for bit in range(10):
      state[1, bit, 5] = bool(self.stage_steps & (1 << bit))
    for bit in range(10):
      state[2, bit, 5] = bool(self.steps & (1 << bit))
    for bit in range(2):
      state[3, bit, 5] = bool(self.steps & (1 << (10 + bit)))
      state[3, 2 + bit, 5] = bool(self.stage_steps & (1 << (10 + bit)))
    state[9, stages.index(self.stage), 5] = True
    state[9, 5 + self.quest, 5] = True
    for bit in range(4):
      state[8, bit, 5] = bool(self.lives & (1 << bit))
      state[8, 4 + bit, 5] = bool(self.glaives & (1 << bit))
    if self.stage == "web":
      for bit in range(7):
        state[7, bit, 5] = bool(self.web_timer & (1 << bit))
    if self.stage == "path":
      state[6, self.path_selection, 5] = True
      state[5, self.correct_path, 5] = True
      for bit in range(5):
        state[4, bit, 5] = bool(self.sun_timer & (1 << bit))
    state[3, 5, 5] = self.lyssa_free if self.stage == "fortress" else False
    state[3, 6, 5] = self.has_fireball if self.stage == "fortress" else False
    return state
