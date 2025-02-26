import pygame
import numpy as np
import random
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Create the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Learning")

# Maze layout (1 = wall, 0 = path)
maze_layout = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,1,1,0,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
]

# Convert maze layout to wall rectangles
walls = []
for y, row in enumerate(maze_layout):
    for x, cell in enumerate(row):
        if cell == 1:
            walls.append(pygame.Rect(x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Agent and redemption point
agent_pos = [GRID_SIZE, GRID_SIZE]
redemption = pygame.Rect(11*GRID_SIZE, 5*GRID_SIZE, GRID_SIZE, GRID_SIZE)

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
actions = ['up', 'down', 'left', 'right']

# Initialize Q-table
Q = defaultdict(lambda: {a: 0 for a in actions})

def get_state():
    return (agent_pos[0] // GRID_SIZE, agent_pos[1] // GRID_SIZE)

def move_agent(action):
    new_pos = agent_pos.copy()
    if action == 'up' and new_pos[1] > 0:
        new_pos[1] -= GRID_SIZE
    elif action == 'down' and new_pos[1] < HEIGHT - GRID_SIZE:
        new_pos[1] += GRID_SIZE
    elif action == 'left' and new_pos[0] > 0:
        new_pos[0] -= GRID_SIZE
    elif action == 'right' and new_pos[0] < WIDTH - GRID_SIZE:
        new_pos[0] += GRID_SIZE
    
    # Check for wall collision
    agent_rect = pygame.Rect(new_pos[0], new_pos[1], GRID_SIZE, GRID_SIZE)
    if not any(agent_rect.colliderect(wall) for wall in walls):
        return new_pos
    return agent_pos

def get_reward():
    agent_rect = pygame.Rect(agent_pos[0], agent_pos[1], GRID_SIZE, GRID_SIZE)
    if agent_rect.colliderect(redemption):
        return 100
    return -0.1

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return max(Q[state], key=Q[state].get)

def update_q_table(state, action, reward, new_state):
    old_value = Q[state][action]
    max_future = max(Q[new_state].values()) if new_state in Q else 0
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * max_future)
    Q[state][action] = new_value

# Main loop
clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # RL Step
    state = get_state()
    action = choose_action(state)
    agent_pos = move_agent(action)
    new_state = get_state()
    reward = get_reward()
    update_q_table(state, action, reward, new_state)
    
    # Print thinking process
    print(f"State: {state}")
    print(f"Possible actions: {Q[state]}")
    print(f"Chosen action: {action}")
    print(f"Reward: {reward}\n")
    
    # Reset position if reached redemption
    if reward == 100:
        agent_pos = [GRID_SIZE, GRID_SIZE]
    
    # Draw elements
    for wall in walls:
        pygame.draw.rect(screen, BLACK, wall)
    pygame.draw.rect(screen, BLUE, redemption)
    pygame.draw.circle(screen, RED, (agent_pos[0] + GRID_SIZE//2, agent_pos[1] + GRID_SIZE//2), GRID_SIZE//2)
    
    pygame.display.flip()
    clock.tick(10)

pygame.quit()