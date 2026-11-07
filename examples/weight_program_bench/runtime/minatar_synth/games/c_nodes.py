from dataclasses import dataclass
import math

import numpy as np


HORIZON = 400
FRAMES_PER_STEP = 4
DIRECTIONS = ((0, -1), (-1, 0), (1, 0), (0, 1))


@dataclass(eq=False)
class Node:
  x: float
  y: float
  vx: float = 0
  vy: float = 0
  anchored: bool = True
  tethered: bool = False


@dataclass
class Link:
  a: Node
  b: Node
  length: float

  def near(self, x: float, y: float) -> bool:
    dx, dy = self.b.x - self.a.x, self.b.y - self.a.y
    px, py = x - self.a.x, y - self.a.y
    squared = dx * dx + dy * dy
    fraction = min(1, max(0, (px * dx + py * dy) / squared)) if squared else 0
    return math.hypot(px - fraction * dx, py - fraction * dy) <= 5


class CNodesEnvironment:
  score_cap = 500

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = HORIZON):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, HORIZON)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = 0
    self.multiplier = 1
    self.next_node = self.boost = 0.0
    self.cursor = (3, 5)
    self.terminal = False
    self.nodes: list[Node] = []
    self.links: list[Link] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    if 1 <= action <= 4:
      dy, dx = DIRECTIONS[action - 1]
      row, col = self.cursor
      self.cursor = min(9, max(2, row + dy)), min(9, max(0, col + dx))
    if action == 5 and self.last_action != 5:
      row, col = self.cursor
      self.links = [link for link in self.links if not link.near(col * 10 + 5, row * 10 + 5)]
      self.multiplier = 1
      self.boost = 1
    self.last_action = action
    self.steps += 1
    reward = 0
    for _ in range(FRAMES_PER_STEP):
      reward += self._advance_frame()
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _advance_frame(self) -> int:
    lowest = max((node.y for node in self.nodes), default=0)
    scroll = 0.03 + self.boost + max(0, 29 - lowest) * 0.1
    self.boost *= 0.7
    self.next_node -= scroll
    if self.next_node < 0:
      new = Node(self.random.uniform(10, 90), 0)
      for node in self.nodes:
        if node.anchored and abs(new.x - node.x) / 2 + abs(node.y) < 30 and self.random.random() < 0.7:
          self.links.append(Link(new, node, math.hypot(new.x - node.x, node.y)))
      self.nodes.append(new)
      self.next_node = 4
    for link in self.links:
      dx, dy = link.b.x - link.a.x, link.b.y - link.a.y
      distance = math.hypot(dx, dy)
      if distance:
        force = (distance - link.length) * 0.02 / distance
        link.a.vx += force * dx
        link.a.vy += force * dy
        link.b.vx -= force * dx
        link.b.vy -= force * dy
    for node in self.nodes:
      node.y += scroll
      node.anchored &= node.y <= 20
      if node.anchored:
        node.vx = node.vy = 0
      else:
        node.x += node.vx
        node.y += node.vy
        node.vx *= 0.95
        node.vy = (node.vy + 0.03) * 0.95
    self.links = [link for link in self.links if min(link.a.y, link.b.y) <= 100]
    self._update_tethers()
    reward = 0
    survivors = []
    for node in self.nodes:
      if node.tethered and node.y > 100:
        self.terminal = True
      if node.y > 140:
        reward += self.multiplier
        self.multiplier = min(64, self.multiplier * 2)
      else:
        survivors.append(node)
    self.nodes = survivors
    alive = set(survivors)
    self.links = [link for link in self.links if link.a in alive and link.b in alive]
    return reward

  def _update_tethers(self) -> None:
    connected = {node: [] for node in self.nodes}
    for link in self.links:
      connected[link.a].append(link.b)
      connected[link.b].append(link.a)
    pending = [node for node in self.nodes if node.anchored]
    reached = set(pending)
    while pending:
      for neighbor in connected[pending.pop()]:
        if neighbor not in reached:
          reached.add(neighbor)
          pending.append(neighbor)
    for node in self.nodes:
      node.tethered = node in reached and not node.anchored

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for node in self.nodes:
      if 0 <= node.x < 100 and 0 <= node.y < 100:
        channel = 0 if node.anchored else 1 if node.tethered else 2
        state[int(node.y) // 10, int(node.x) // 10, channel] = True
    for link in self.links:
      for fraction in np.linspace(0, 1, 12):
        x = link.a.x + (link.b.x - link.a.x) * fraction
        y = link.a.y + (link.b.y - link.a.y) * fraction
        if 0 <= x < 100 and 0 <= y < 100:
          state[int(y) // 10, int(x) // 10, 3] = True
    row, col = self.cursor
    state[row, col, 4] = True
    for row, value in enumerate((self.steps, self.multiplier, self.last_action)):
      for bit in range(9):
        state[row, bit, 5] = bool(value & (1 << bit))
    return state
