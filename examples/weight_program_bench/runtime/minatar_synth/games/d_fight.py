from dataclasses import dataclass
import math

import numpy as np


@dataclass(eq=False)
class Agent:
  x: float
  y: float = 87
  team: int = 0
  vx: float = 0
  vy: float = 0
  spacing: float = 45
  target: "Agent | None" = None
  aim: float = 0
  fire: float = 0


class DFightEnvironment:
  score_cap = 25

  def __init__(self, sticky_action_prob: float = 0.1, horizon: int = 400):
    self.sticky_action_prob = sticky_action_prob
    self.horizon = min(horizon, 400)
    self.seed(0)
    self.reset()

  def seed(self, seed: int) -> None:
    self.random = np.random.default_rng(seed)

  def reset(self) -> None:
    self.steps = self.last_action = self.enemy_timer = 0
    self.ally_timer, self.dome_timer = 9, 0.0
    self.terminal = False
    self.player = Agent(99)
    self.agents = [self.player]
    self.domes = [100.0]
    self.shots: list[tuple[float, float, int]] = []

  @classmethod
  def normalize_score(cls, score: float) -> float:
    return min(max(score, 0), cls.score_cap) / cls.score_cap

  def act(self, action: int) -> tuple[float, bool]:
    if self.terminal:
      return 0.0, True
    if self.random.random() < self.sticky_action_prob:
      action = self.last_action
    self.last_action = action
    self.steps += 1
    direction = -1 if action == 1 else 1 if action == 4 else 0
    reward = 0
    for _ in range(4):
      reward += self._advance_frame(direction)
      if self.terminal:
        break
    self.terminal |= self.steps >= self.horizon
    return float(reward), self.terminal

  def _spawn(self, team: int) -> None:
    left = self.random.random() < 0.3
    self.agents.append(Agent(-3 if left else 203, team=team, vx=1 if left else -1,
                             spacing=self.random.uniform(30, 60)))

  def _advance_frame(self, direction: int) -> int:
    self.dome_timer -= 0.2
    if self.dome_timer < 0:
      self.domes.append(240)
      self.dome_timer += self.random.uniform(70, 150)
    self.domes = [x - 0.2 for x in self.domes if x > -40]
    enemy_count = sum(agent.team == 1 for agent in self.agents)
    ally_count = len(self.agents) - enemy_count - 1
    self.ally_timer -= 1
    self.enemy_timer = self.enemy_timer - 1 if enemy_count else -1
    if self.ally_timer < 0:
      self._spawn(0)
      self.ally_timer = int(self.random.integers(120, 151)) * (ally_count + 1)
    if self.enemy_timer < 0:
      self._spawn(1)
      self.enemy_timer = int(self.random.integers(60, 81)) * (enemy_count + 1)

    self.player.x = min(197, max(3, self.player.x + 2 * direction))
    self.shots = []
    for agent in self.agents:
      if agent is not self.player:
        self._move(agent)
      if agent.target not in self.agents:
        agent.target = min(
          (other for other in self.agents if other.team != agent.team),
          key=lambda other: math.hypot(other.x - agent.x, other.y - agent.y),
          default=None,
        )
        agent.aim = agent.fire = 0
      if agent.target is not None:
        self._aim(agent)

    survivors = []
    reward = 0
    for agent in self.agents:
      hit = any(team != agent.team and abs(x - agent.x) < 6 and abs(y - agent.y) < 5
                for x, y, team in self.shots)
      if hit and agent is self.player:
        self.terminal = True
      if hit and agent.team == 1:
        reward += 1
      if not hit or agent is self.player:
        survivors.append(agent)
    self.agents = survivors
    return reward

  def _move(self, agent: Agent) -> None:
    agent.vy += 0.05
    if (agent.x < 3 and agent.vx < 0) or (agent.x > 197 and agent.vx > 0):
      agent.x = min(197, max(3, agent.x))
      agent.vx *= -1
    if agent.y >= 87:
      agent.y = 87
      agent.vy = -self.random.uniform(3, 6) if self.random.random() < 0.01 else 0
    agent.x += agent.vx
    agent.y += agent.vy
    agent.vx *= 0.9
    agent.vy *= 0.9

  def _aim(self, agent: Agent) -> None:
    target = agent.target
    dx, dy = target.x - agent.x, target.y - agent.y
    distance = math.hypot(dx, dy)
    if distance > 99:
      agent.target = None
      agent.aim = agent.fire = 0
      return
    if agent is not self.player:
      agent.vx += math.copysign(0.1, dx) * (1 if distance > agent.spacing else -1)
    if agent.fire:
      agent.fire += 3
      ratio = agent.fire / max(distance, 0.01)
      x, y = agent.x + dx * ratio, agent.y + dy * ratio
      blocked = any(abs(math.hypot(x - dome, y - 90) - 40) < 4 for dome in self.domes)
      if blocked or y > 90 or agent.fire > 99:
        agent.fire = agent.aim = 0
      else:
        self.shots.append((x, y, agent.team))
    else:
      rate = 0.9 if agent is self.player else 0.4 if agent.team else 0.2
      agent.aim += rate * max(1, distance) * 0.03
      if agent.aim > distance:
        agent.fire = 5

  def state(self) -> np.ndarray:
    state = np.zeros((10, 10, 6), dtype=bool)
    for agent in self.agents:
      if 0 <= agent.x < 200 and 0 <= agent.y < 100:
        state[int(agent.y) // 10, int(agent.x) // 20,
              0 if agent is self.player else 2 if agent.team else 1] = True
    for x, y, team in self.shots:
      if 0 <= x < 200 and 0 <= y < 100:
        state[int(y) // 10, int(x) // 20, 3 + team] = True
    for dome in self.domes:
      for angle in np.linspace(math.pi, math.tau, 20):
        x, y = dome + 40 * math.cos(angle), 90 + 40 * math.sin(angle)
        if 0 <= x < 200:
          state[int(y) // 10, int(x) // 20, 5] = True
    for bit in range(9):
      state[0, bit, 5] = bool(self.steps & (1 << bit))
    return state
