import numpy as np


HORIZON = 500
UNIT = 8
ANCHOR_SPACING = 16
BOTTOM = 80
REACH = 16


class GrapplingHEnvironment:
  # At most four scroll units per step, plus one rounding point per catch.
  score_cap = HORIZON + HORIZON * 4 // UNIT

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.x, self.y = 24, 16
    self.vx = self.vy = 0
    self.steps = self.scroll = self.since_release = self.last_action = 0
    self.anchors = [(40, 0), (48, 8)]
    self.attached: tuple[int, int] | None = self.anchors[0]
    self.rearmed = False
    self.terminal = False

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def _nearby(self) -> tuple[int, int] | None:
    candidates = [(dx * dx + dy * dy, anchor)
                  for anchor in self.anchors
                  for dx, dy in [(anchor[0] - self.x, anchor[1] + self.scroll - self.y)]]
    distance, anchor = min(candidates, default=(REACH ** 2, None))
    return anchor if distance < REACH ** 2 else None

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    pressed = action == 5 and self.last_action != 5
    self.last_action = action
    self.steps += 1
    reward = 0
    if self.attached is None and action == 5 and self.rearmed:
      self.attached = self._nearby()
      if self.attached is not None:
        reward = (self.since_release + UNIT - 1) // UNIT
        pressed = False

    advance = max(1, min(4, (40 - self.y) // 8))
    previous_scroll = self.scroll
    self.scroll += advance
    self.since_release += advance
    self.y += advance
    if self.scroll // ANCHOR_SPACING > previous_scroll // ANCHOR_SPACING:
      self.anchors.append((UNIT * int(self.random.integers(1, 9)),
                           -(self.scroll // ANCHOR_SPACING) * ANCHOR_SPACING))

    self.vy += 1
    if self.attached is not None:
      dx = self.attached[0] - self.x
      dy = self.attached[1] + self.scroll - self.y
      divisor = 4 if pressed else 12
      self.vx += round(dx / divisor)
      self.vy += round(dy / divisor)
      if pressed:
        self.anchors.remove(self.attached)
        self.attached = None
        self.since_release = 0
        self.rearmed = False
    self.vx = max(-12, min(12, self.vx))
    self.vy = max(-12, min(12, self.vy))
    self.x += self.vx
    self.y = max(0, self.y + self.vy)
    if not 0 <= self.x < BOTTOM:
      self.x = max(0, min(BOTTOM - 1, self.x))
      self.vx = -round(self.vx * 0.8)
    expired = self.attached is not None and self.attached[1] + self.scroll >= BOTTOM
    self.anchors = [anchor for anchor in self.anchors if anchor[1] + self.scroll < BOTTOM]
    if action != 5:
      self.rearmed = True
    self.terminal = expired or self.y >= BOTTOM or self.steps >= self.horizon
    return float(reward), self.terminal

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, self.y // UNIT), self.x // UNIT, 0] = True
    nearby = self._nearby() if self.attached is None else None
    for anchor in self.anchors:
      x, y = anchor
      row, column = (y + self.scroll) // UNIT, x // UNIT
      state[row, column, 2] = True
      if anchor == self.attached:
        state[row, column, 3] = True
      if anchor == nearby:
        state[row, column, 4] = True
    values = (self.x, self.y, self.vx + 12, self.vy + 12, self.scroll % ANCHOR_SPACING,
              self.since_release, self.steps, int(self.rearmed), int(self.last_action == 5))
    for row, value in enumerate(values):
      for bit in range(10):
        state[row, bit, 5] = bool(value & (1 << bit))
    state[9, 0, 5] = bool(self.since_release & 1024)
    return state
