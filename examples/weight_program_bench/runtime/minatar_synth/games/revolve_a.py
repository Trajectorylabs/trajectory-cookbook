from math import cos, pi, sin

import numpy as np


class RevolveAEnvironment:
  score_cap = 200

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.position = np.array([50.0, 50.0])
    self.velocity = np.full(2, 2 ** -0.5)
    self.angle = -pi / 2
    self.steps = self.timer = 0
    self.held = self.terminal = False
    self.enemies = np.empty((0, 2))
    self.velocities = np.empty((0, 2))
    self.removed = np.empty(0, dtype=bool)

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    held = self.held if self.random.random() < self.sticky_action_prob else action == 5
    tap = held and not self.held
    self.held = held
    self.steps += 1
    reward = 0
    for frame in range(4):
      reward += self._advance(tap and frame == 0)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self):
    position = np.array([self.random.uniform(0, 99), self.random.choice((-3, 103))])
    if self.random.random() < 0.5:
      position = position[::-1]
    delta = self.random.uniform(10, 90, 2) - position
    self.enemies = np.vstack((self.enemies, position))
    self.velocities = np.vstack((self.velocities, 0.3 * delta / np.linalg.norm(delta)))
    self.removed = np.append(self.removed, False)

  def _advance(self, tap):
    self.timer -= 1
    if self.timer < 0:
      self._spawn()
      self.timer = self.random.uniform(30, 40)
    count = int(self.removed.sum())
    self.enemies[~self.removed] += self.velocities[~self.removed]
    keep = ~self.removed & np.all((self.enemies >= -5) & (self.enemies <= 105), axis=1)
    self.enemies = self.enemies[keep]
    self.velocities = self.velocities[keep]
    self.removed = np.zeros(len(self.enemies), dtype=bool)
    if tap:
      self.velocity[:] = cos(self.angle), sin(self.angle)
    bounce = ((self.position < 3) & (self.velocity < 0)) | ((self.position > 97) & (self.velocity > 0))
    self.velocity[bounce] *= -1
    self.position += self.velocity * 0.4
    self.angle = (self.angle + 0.08) % (2 * pi)
    delta = self.enemies - self.position
    self.terminal = bool(np.any(np.all(abs(delta) < 4.5, axis=1)))
    if not self.terminal:
      links = self._links()
      if len(links):
        starts, ends = self.enemies[links[:, 0]], self.enemies[links[:, 1]]
        vectors = ends - starts
        lengths = np.sum(vectors ** 2, axis=1)
        projected = np.divide(np.sum((self.position - starts) * vectors, axis=1),
                              lengths, out=np.zeros_like(lengths), where=lengths > 0)
        closest = starts + np.clip(projected, 0, 1)[:, None] * vectors
        if np.any(np.linalg.norm(closest - self.position, axis=1) < 3):
          self.removed = np.linalg.norm(delta, axis=1) < 30
          while True:
            touching = self.removed[links].any(axis=1)
            linked = links[touching].ravel()
            if self.removed[linked].all():
              break
            self.removed[linked] = True
    return count * (count + 1) // 2

  def _links(self):
    left, right = np.triu_indices(len(self.enemies), 1)
    near = np.linalg.norm(self.enemies[left] - self.enemies[right], axis=1) < 30
    return np.column_stack((left[near], right[near]))

  @staticmethod
  def _mark(state, points, channel):
    visible = np.all((points >= 0) & (points < 100), axis=1)
    cols, rows = (points[visible] // 10).astype(int).T
    state[rows, cols, channel] = True

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    self._mark(state, self.position[None], 0)
    self._mark(state, self.enemies[~self.removed], 1)
    for left, right in self._links():
      if not (self.removed[left] or self.removed[right]):
        self._mark(state, np.linspace(self.enemies[left], self.enemies[right], 31), 2)
    aim = self.position + 10 * np.array([cos(self.angle), sin(self.angle)])
    self._mark(state, np.clip(aim, 0, 99)[None], 3)
    self._mark(state, self.enemies[self.removed], 4)
    for row, value in enumerate((self.steps, self.held, round(self.angle * 50),
                                 round((self.velocity[0] + 1) * 100),
                                 round((self.velocity[1] + 1) * 100))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
