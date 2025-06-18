import pygame
import math
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 800
center = np.array([WIDTH // 2, HEIGHT // 2])
outer_radius = 300
inner_radius = 230
FINISH_LINE_ANGLE = math.pi
FINISH_LINE_WIDTH = math.radians(20)

player_radius = 5
player_speed = 3

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Training Circular RL Agent")

class CircularRoadEnv:
    def __init__(self):
        self.center = center
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.player_radius = player_radius
        self.player_speed = player_speed
        self.FINISH_LINE_ANGLE = FINISH_LINE_ANGLE
        self.FINISH_LINE_WIDTH = FINISH_LINE_WIDTH
        self.dead = False
        self.reset()

    def _get_state(self):
        dx = self.player_pos[0] - self.center[0]
        dy = self.player_pos[1] - self.center[1]
        dist = np.linalg.norm([dx, dy])
        angle = math.atan2(dy, dx)
        angle_to_finish = (self.FINISH_LINE_ANGLE - angle) % (2 * math.pi)
        if angle_to_finish > math.pi:
            angle_to_finish -= 2 * math.pi

        dist_to_outer = self.outer_radius - dist
        dist_to_inner = dist - self.inner_radius

        return np.array([
            dx / self.outer_radius,
            dy / self.outer_radius,
            dist / self.outer_radius,
            math.cos(angle_to_finish),
            math.sin(angle_to_finish),
            dist_to_outer / self.outer_radius,
            dist_to_inner / self.outer_radius
        ], dtype=np.float32)

    def reset(self):
        start_radius = (self.inner_radius + self.outer_radius) / 2
        self.player_pos = np.array([
            self.center[0] + start_radius * math.cos(0),
            self.center[1] + start_radius * math.sin(0)
        ], dtype=np.float32)
        self.dead = False
        return self._get_state()

    def step(self, action: np.ndarray):
        if self.dead:
            return self._get_state(), 0.0, True

        dx = (action[1] - action[0]) * self.player_speed
        dy = (action[3] - action[2]) * self.player_speed
        new_pos = self.player_pos + np.array([dx, dy], dtype=np.float32)

        dist = np.linalg.norm(new_pos - self.center)
        collision = (dist + self.player_radius > self.outer_radius) or \
                    (dist - self.player_radius < self.inner_radius)

        angle = math.atan2(new_pos[1] - self.center[1], new_pos[0] - self.center[0])
        at_finish = abs((angle - self.FINISH_LINE_ANGLE) % (2 * math.pi)) < self.FINISH_LINE_WIDTH / 2

        if collision:
            reward = -10.0
            self.dead = True
        elif at_finish and self.inner_radius < dist < self.outer_radius:
            reward = 100.0
            self.dead = True
        else:
            reward = 0.05 + 0.1 * (1 - abs(angle - self.FINISH_LINE_ANGLE) / math.pi)
            self.player_pos = new_pos

        return self._get_state(), reward, self.dead

    def render(self):
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (50, 50, 50), self.center.astype(int), self.outer_radius)
        pygame.draw.circle(screen, (30, 30, 30), self.center.astype(int), self.inner_radius)

        pygame.draw.arc(
            screen, (0, 150, 255),
            (self.center[0] - self.outer_radius, self.center[1] - self.outer_radius,
             self.outer_radius * 2, self.outer_radius * 2),
            self.FINISH_LINE_ANGLE - self.FINISH_LINE_WIDTH / 2,
            self.FINISH_LINE_ANGLE + self.FINISH_LINE_WIDTH / 2,
            5
        )

        color = (255, 0, 0) if self.dead else (0, 200, 50)
        pygame.draw.circle(screen, color, self.player_pos.astype(int), self.player_radius)
        pygame.display.flip()
