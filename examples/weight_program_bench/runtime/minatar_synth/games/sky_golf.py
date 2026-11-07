from math import cos, hypot, pi, sin

import numpy as np


FAIRWAY, SAND, WATER, TREE, FLAG = range(5)


class SkyGolfEnvironment:
  score_cap = 3
  hole_seeds = (71, 45, 9)

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.hole = self.ammo = self.steps = 0
    self.held = self.terminal = False
    self._start_hole()

  def _start_hole(self):
    course = np.random.default_rng(self.hole_seeds[self.hole])
    self.tiles = []

    def platform(x, y, count, tee=False, flag=False):
      kinds = np.full(count, FAIRWAY)
      heights = np.zeros(count)
      for kind, probability in ((TREE, 1), (FAIRWAY, 0.7), (SAND, 0.8), (WATER, 0.5)):
        if course.random() < probability:
          width = int(course.integers(3, count // 2))
          start = int(course.integers(0, count - width))
          kinds[start:start + width] = kind
          if kind == TREE:
            heights[start:start + width] = course.integers(10, 20) + course.integers(-5, 6, width)
      if tee:
        kinds[:5] = FAIRWAY
      if flag:
        width = int(course.integers(3, 6))
        start = count - width - int(course.integers(3))
        kinds[start:start + width] = FAIRWAY
        kinds[start + width // 2] = FLAG
      self.tiles.extend((x + i * 6, y, kind, heights[i]) for i, kind in enumerate(kinds))

    width, y = 25, 90
    count = course.integers(1, 3)
    if course.random() < 0.5:
      width = course.integers(12, 20)
      if course.random() < 0.5:
        y = course.integers(30, 70)
    platform(0, y, width, tee=True)
    for index in range(count):
      width = course.integers(9, 20)
      platform(course.integers(150 - 6 * width), course.integers(30, 70),
               width, flag=index == count - 1)
    self.x, self.y = 5., y - 7.
    self.vx = self.vy = self.angle = 0.
    self.direction = -1
    self.base = 1.
    self.previous = self.x, self.y, self.base
    self.power = 0.1
    self.mode = "aim"
    self.ammo += 5

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap, release = held and not self.held, self.held and not held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0, release and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _surface(self, x, y):
    for left, bottom, kind, height in self.tiles:
      if left - 2 < x < left + 8 and bottom - 7 < y < bottom + 2:
        return kind
      if kind == TREE and left - 1 < x < left + 7 and bottom - 7 - height < y < bottom - 5:
        return FAIRWAY
    return None

  def _advance(self, tap, release):
    if self.mode == "aim":
      self.angle += self.direction * 0.05
      if not -pi <= self.angle <= 0:
        self.direction *= -1
        self.angle += self.direction * 0.1
      if tap:
        self.mode = "power"
    elif self.mode == "power":
      self.power += 0.2
      if self.power > 9 or release:
        speed = self.power * 0.5 * self.base
        self.vx, self.vy = speed * cos(self.angle), speed * sin(self.angle)
        self.mode = "flight"
        self.ammo -= 1
    else:
      return self._fly()
    return 0

  def _fly(self):
    if self._surface(self.x + self.vx, self.y) is not None or not 2 <= self.x + self.vx <= 148:
      self.vx *= -0.8
      self.vy *= 0.8
    surface = self._surface(self.x, self.y + self.vy)
    if surface is not None:
      factor = {SAND: 0.5, WATER: 0.4}.get(surface, 0.8) if self.vy > 0 else 0.8
      self.vy *= -factor
      self.vx *= factor
      if self.vy < 0 and hypot(self.vx, self.vy) < 0.5:
        if surface == FLAG:
          self.hole += 1
          self.terminal = self.hole == len(self.hole_seeds)
          if not self.terminal:
            self._start_hole()
          return 1
        if surface == WATER:
          self.x, self.y, self.base = self.previous
        else:
          self.base = 0.5 if surface == SAND else 1
          self.previous = self.x, self.y, self.base
        self._rest()
        return 0
    self.x += self.vx
    self.y += self.vy
    self.vx *= 0.98
    self.vy = self.vy * 0.98 + 0.1
    if self.y > 110:
      self.x, self.y, self.base = self.previous
      self._rest()
    return 0

  def _rest(self):
    self.mode = "aim"
    self.power = 0.1
    self.terminal = self.ammo <= 0

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[int(np.clip(self.y // 10, 0, 9)), int(np.clip(self.x // 15, 0, 9)), 0] = True
    for x, y, kind, height in self.tiles:
      col = int(x // 15)
      channel = 2 if kind == SAND else 3 if kind == WATER else 4 if kind == FLAG else 1
      state[int((y - 3) // 10), col, channel] = True
      if kind == TREE:
        state[int((y - 5 - height) // 10):int((y - 5) // 10) + 1, col, 1] = True
    gauges = (self.steps, self.hole, self.ammo, round(self.power * 50),
              round(-self.angle * 100), self.direction > 0, self.held,
              ("aim", "power", "flight").index(self.mode), self.base < 1)
    for row, value in enumerate(gauges):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
