import numpy as np


TARGET_SCORES = {
  "plane": 200,
  "fuel": 300,
  "turret": 400,
  "mobot": 400,
  "dodger": 500,
  "radar": 1_000,
}
BARRIERS = ((20, 3), (55, 1), (110, 2), (145, 3))


class ZaxxonEnvironment:
  """Four finite fortress-space-fortress-boss rounds with altitude and fuel."""

  score_cap = 54_400.0

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
    self.round = 0
    self.lives = 5
    self.score = 0.0
    self.steps = 0
    self.last_action = 0
    self.bonus_fighter_awarded = False
    self.terminal = False
    self._start_round()

  def _start_round(self) -> None:
    self.progress = 0
    self.lane = 1
    self.altitude = 3
    self.fuel = 100
    self.bullets: list[list[int]] = []
    self.enemy_shots: list[list[int]] = []
    self.targets = self._targets()
    self.boss_hp = 3
    self.boss_timer = 24
    self.boss_missile = 0

  def _targets(self) -> list[dict[str, int | str]]:
    fortress = (
      (8, 0, 0, "fuel"),
      (14, 2, 0, "turret"),
      (27, 1, 0, "mobot"),
      (34, 0, 0, "fuel"),
      (41, 2, 1, "turret"),
      (47, 1, 2, "mobot"),
      (51, 0, 1, "dodger"),
      (58, 2, 0, "radar"),
    )
    targets = []
    identifier = 0
    for offset in (0, 90):
      for progress, lane, altitude, kind in fortress:
        targets.append({"id": identifier, "progress": progress + offset, "lane": lane, "altitude": altitude, "kind": kind})
        identifier += 1
    for index in range(6):
      targets.append({"id": identifier, "progress": 65 + index * 4, "lane": index % 3, "altitude": 1 + index % 4, "kind": "plane"})
      identifier += 1
    return targets

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.rand() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    if action == 1:
      self.lane = max(0, self.lane - 1)
    elif action == 2:
      self.altitude = min(4, self.altitude + 1)
    elif action == 3:
      self.lane = min(2, self.lane + 1)
    elif action == 4:
      self.altitude = max(0, self.altitude - 1)
    elif action == 5:
      self.bullets.append([self.progress, self.lane, self.altitude])
    reward = self._advance_world()
    self.score += reward
    if self.score >= 10_000 and not self.bonus_fighter_awarded:
      self.lives += 1
      self.bonus_fighter_awarded = True
    self.steps += 1
    if self.steps >= self.horizon:
      self.terminal = True
    return reward, self.terminal

  def _advance_world(self) -> float:
    if self.progress >= 150:
      return self._boss_step()
    previous_progress = self.progress
    self.progress += 1
    self.fuel -= 1
    for bullet in self.bullets:
      bullet[0] += 2
    reward = self._resolve_target_hits(previous_progress)
    self._move_enemy_shots()
    if any(self.progress == barrier and self.altitude != opening for barrier, opening in BARRIERS):
      self._lose_fighter()
    collision = any(
      int(target["progress"]) == self.progress
      and int(target["lane"]) == self.lane
      and int(target["altitude"]) == self.altitude
      for target in self.targets
    )
    collision |= any(shot == [self.progress, self.lane, self.altitude] for shot in self.enemy_shots)
    if collision or self.fuel <= 0:
      self._lose_fighter()
    self._spawn_enemy_shot()
    return reward

  def _resolve_target_hits(self, previous_progress: int) -> float:
    reward = 0.0
    hit_ids: set[int] = set()
    used_bullets: set[int] = set()
    for bullet_index, bullet in enumerate(self.bullets):
      for target in self.targets:
        target_progress = int(target["progress"])
        if not previous_progress < target_progress <= bullet[0]:
          continue
        if bullet[1] != target["lane"] or bullet[2] != target["altitude"]:
          continue
        hit_ids.add(int(target["id"]))
        used_bullets.add(bullet_index)
        kind = str(target["kind"])
        reward += TARGET_SCORES[kind]
        if kind == "fuel":
          self.fuel = min(100, self.fuel + 30)
        break
    self.targets = [target for target in self.targets if int(target["id"]) not in hit_ids]
    self.bullets = [bullet for index, bullet in enumerate(self.bullets) if index not in used_bullets and bullet[0] <= self.progress + 10]
    return float(reward)

  def _spawn_enemy_shot(self) -> None:
    shooters = [target for target in self.targets if target["kind"] in ("turret", "mobot", "dodger") and 0 < int(target["progress"]) - self.progress <= 8]
    if shooters and self.steps % 5 == 0:
      source = shooters[0]
      self.enemy_shots.append([int(source["progress"]), int(source["lane"]), int(source["altitude"])])

  def _move_enemy_shots(self) -> None:
    for shot in self.enemy_shots:
      shot[0] -= 1
    self.enemy_shots = [shot for shot in self.enemy_shots if shot[0] >= self.progress]

  def _boss_step(self) -> float:
    reward = 0.0
    aligned_bullets = [bullet for bullet in self.bullets if bullet[1] == 1 and bullet[2] == 2]
    if aligned_bullets and self.boss_missile == 0:
      self.bullets.remove(aligned_bullets[0])
      self.boss_hp -= 1
      if self.boss_hp == 0:
        reward = 5_000.0
        self._advance_round()
      return reward
    self.boss_timer -= 1
    if self.boss_timer == 0:
      self.boss_missile = 5
    if self.boss_missile:
      if aligned_bullets:
        self.bullets.remove(aligned_bullets[0])
        reward = 1_000.0
        self._advance_round()
      else:
        self.boss_missile -= 1
        if self.boss_missile == 0:
          self._lose_fighter()
          self.boss_timer = 24
    return reward

  def _advance_round(self) -> None:
    self.round += 1
    if self.round == 4:
      self.terminal = True
    else:
      self._start_round()

  def _lose_fighter(self) -> None:
    self.lives -= 1
    if self.lives <= 0:
      self.terminal = True
      return
    self.lane = 1
    self.altitude = 3
    self.fuel = max(40, self.fuel)
    self.bullets.clear()
    self.enemy_shots.clear()

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[8, 2 + self.lane * 3, 0] = True
    state[9 - self.altitude, 2 + self.lane * 3, 4] = True
    for target in self.targets:
      distance = int(target["progress"]) - self.progress
      if 0 <= distance < 9:
        channel = 1 if target["kind"] == "plane" else 2
        state[8 - distance, 2 + int(target["lane"]) * 3, channel] = True
    for bullet in self.bullets:
      distance = bullet[0] - self.progress
      if 0 <= distance < 9:
        state[8 - distance, 2 + bullet[1] * 3, 3] = True
    for shot in self.enemy_shots:
      distance = shot[0] - self.progress
      if 0 <= distance < 9:
        state[8 - distance, 2 + shot[1] * 3, 1] = True
    values = ((self.round, 3), (self.progress, 8), (self.altitude, 3), (self.lane, 2), (self.fuel, 7), (self.lives, 4), (self.boss_hp, 2), (self.boss_timer, 5))
    offset = 0
    for value, width in values:
      for bit in range(width):
        state[offset // 10, offset % 10, 5] = bool(value & (1 << bit))
        offset += 1
    state[4, 0, 5] = self.progress >= 150
    state[4, 1, 5] = self.boss_missile > 0
    return state
