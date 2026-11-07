from dataclasses import dataclass

import numpy as np


@dataclass
class Block:
  x: float
  y: float
  fixed: bool = False


class PileupEnvironment:
  score_cap = 2500

  def __init__(self, sticky_action_prob=0.1, horizon=400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed):
    self.random = np.random.default_rng(seed)

  def reset(self):
    self.x, self.y = 50.0, 80.0
    self.board: Block | None = None
    self.board_speed = 0.0
    self.blocks: list[Block] = []
    self.timer = self.steps = self.action = 0
    self.terminal = False

  @classmethod
  def normalize_score(cls, score):
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action):
    if self.terminal:
      return 0.0, True
    if self.random.random() >= self.sticky_action_prob:
      self.action = action
    self.steps += 1
    reward = 0.0
    for _ in range(4):
      reward += self._advance()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return reward, self.terminal

  def _board_hits(self, x, y):
    return self.board is not None and abs(self.board.x - x) < 13.5 and abs(self.board.y - y) < 6

  def _advance(self):
    reward = 0.0
    if self.board is None:
      self.board = Block(self.random.uniform(0, 99), 0)
      self.board_speed = 0.0
    else:
      self.board_speed += 0.1
      self.board.y += self.board_speed
      if self.board.y > 99:
        self.board = None
        reward = 99 - self.y
    y = 80
    remaining = []
    for block in self.blocks:
      if block.fixed:
        y = min(y, block.y - 9)
      else:
        block.y += 1
        block.fixed = block.y + 3.5 > 95 or any(
          other.fixed and abs(block.x - other.x) < 7 and abs(block.y - other.y) < 7
          for other in remaining
        )
      if not self._board_hits(block.x, block.y):
        remaining.append(block)
    self.blocks = remaining
    movement = (self.action == 3) - (self.action == 1)
    self.x = min(99, max(0, self.x + 2 * movement))
    self.y = y
    self.terminal = y < 0 or self._board_hits(self.x, y)
    self.timer -= 1
    if self.timer < 0:
      self.blocks.append(Block(self.x, y))
      self.timer += 10
    return reward

  def state(self):
    state = np.zeros((10, 10, 6), dtype=bool)
    state[min(9, max(0, int(self.y // 10))), int(self.x // 10), 0] = True
    for block in self.blocks:
      if 0 <= block.y < 100:
        state[int(block.y // 10), int(block.x // 10), 1 if block.fixed else 2] = True
    if self.board is not None:
      left = max(0, int((self.board.x - 10) // 10))
      right = min(9, int((self.board.x + 10) // 10))
      state[int(self.board.y // 10), left:right + 1, 3] = True
    state[9, :, 4] = True
    for row, value in enumerate((self.steps, self.timer, round(self.board_speed * 100), round(self.x * 4))):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
