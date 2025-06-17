import pygame
import math
import json
import time
import os
from pathlib import Path

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Circular Road with Data Logging")

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Colors
BG_COLOR = (30, 30, 30)
ROAD_COLOR = (50, 50, 50)
PLAYER_COLOR = (0, 200, 50)
PLAYER_DEAD_COLOR = (255, 0, 0)
FINISH_COLOR = (0, 150, 255)

# Game geometry
center = (WIDTH // 2, HEIGHT // 2)
outer_radius = 300
inner_radius = 230
FINISH_LINE_ANGLE = math.pi  # 180 degrees (bottom)
FINISH_LINE_WIDTH = math.radians(20)  # Convert to radians

# Colliders (invisible)
collider_radius = 2
num_colliders = 180
collider_circles = []

for i in range(num_colliders):
    angle = (2 * math.pi / num_colliders) * i
    collider_circles.append((
        center[0] + outer_radius * math.cos(angle),
        center[1] + outer_radius * math.sin(angle)
    ))
    collider_circles.append((
        center[0] + inner_radius * math.cos(angle),
        center[1] + inner_radius * math.sin(angle)
    ))

# Player setup
player_radius = 5
start_radius = (inner_radius + outer_radius) / 2
start_pos = [
    center[0] + start_radius * math.cos(0),
    center[1] + start_radius * math.sin(0)
]
player_pos = start_pos[:]
player_speed = 3

# Game state
dead = False
death_timer = 0
DEATH_DURATION = 60
episode_reward = 0
episode_start_time = time.time()

class DataLogger:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "timestamps": [],
            "metadata": {
                "episode_start": time.time(),
                "player_speed": player_speed,
                "map_geometry": {
                    "outer_radius": outer_radius,
                    "inner_radius": inner_radius
                }
            }
        }
    
    def log_step(self, state, action, reward, done):
        self.episode_data["states"].append(state)
        self.episode_data["actions"].append(action)
        self.episode_data["rewards"].append(reward)
        self.episode_data["dones"].append(done)
        self.episode_data["timestamps"].append(time.time() - self.episode_data["metadata"]["episode_start"])
        
        if done:
            self.save_episode()
            self.reset()
    
    def save_episode(self):
        filename = DATA_DIR / f"episode_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(self.episode_data, f, indent=2)
        print(f"Saved episode data to {filename}")

data_logger = DataLogger()

def get_state():
    """Normalized state representation for RL"""
    dist = math.hypot(player_pos[0] - center[0], player_pos[1] - center[1])
    angle = math.atan2(player_pos[1] - center[1], player_pos[0] - center[0])
    angle_to_finish = (FINISH_LINE_ANGLE - angle) % (2 * math.pi)
    if angle_to_finish > math.pi:
        angle_to_finish -= 2 * math.pi
    
    return [
        (player_pos[0] - center[0]) / outer_radius,  # Normalized X
        (player_pos[1] - center[1]) / outer_radius,  # Normalized Y
        dist / outer_radius,                         # Distance ratio
        math.cos(angle_to_finish),                   # Finish direction X
        math.sin(angle_to_finish)                    # Finish direction Y
    ]

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(60)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Action capture (one-hot encoded)
    keys = pygame.key.get_pressed()
    action = [
        int(keys[pygame.K_LEFT] or keys[pygame.K_a]),
        int(keys[pygame.K_RIGHT] or keys[pygame.K_d]),
        int(keys[pygame.K_UP] or keys[pygame.K_w]),
        int(keys[pygame.K_DOWN] or keys[pygame.K_s])
    ]
    
    if not dead:
        # Movement
        dx = (action[1] - action[0]) * player_speed
        dy = (action[3] - action[2]) * player_speed
        new_pos = [player_pos[0] + dx, player_pos[1] + dy]
        
        # Collision check
        dist = math.hypot(new_pos[0] - center[0], new_pos[1] - center[1])
        collision = (dist + player_radius > outer_radius) or (dist - player_radius < inner_radius)
        
        # Finish line check
        angle = math.atan2(new_pos[1] - center[1], new_pos[0] - center[0])
        at_finish = (
            abs((angle - FINISH_LINE_ANGLE) % (2 * math.pi)) < FINISH_LINE_WIDTH/2
            and inner_radius < dist < outer_radius
        )
        
        # Reward calculation
        if collision:
            reward = -10
            dead = True
            death_timer = DEATH_DURATION
        elif at_finish:
            reward = 100
            dead = True  # Episode success
        else:
            reward = 0.1 * (1 - abs(angle - FINISH_LINE_ANGLE)/math.pi)  # Progress reward
            player_pos = new_pos
        
        episode_reward += reward
        data_logger.log_step(get_state(), action, reward, dead)
    
    else:
        # Death/respawn handling
        death_timer -= 1
        if death_timer <= 0:
            dead = False
            player_pos = start_pos[:]
            print(f"Episode reward: {episode_reward:.1f}")
            episode_reward = 0
    
    # Rendering
    screen.fill(BG_COLOR)
    pygame.draw.circle(screen, ROAD_COLOR, center, outer_radius)
    pygame.draw.circle(screen, BG_COLOR, center, inner_radius)
    
    # Draw finish line
    pygame.draw.arc(
        screen, FINISH_COLOR,
        (center[0] - outer_radius, center[1] - outer_radius, outer_radius*2, outer_radius*2),
        FINISH_LINE_ANGLE - FINISH_LINE_WIDTH/2,
        FINISH_LINE_ANGLE + FINISH_LINE_WIDTH/2,
        5
    )
    
    # Draw player
    color = PLAYER_DEAD_COLOR if dead and (death_timer // 5) % 2 == 0 else PLAYER_COLOR
    pygame.draw.circle(screen, color, (int(player_pos[0]), int(player_pos[1])), player_radius)
    
    pygame.display.flip()

pygame.quit()