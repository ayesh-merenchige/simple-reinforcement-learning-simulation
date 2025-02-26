import numpy as np
import pygame
import random
import time
import sys

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Maze dimensions
CELL_SIZE = 40
MAZE_WIDTH = 10
MAZE_HEIGHT = 10
WIDTH = MAZE_WIDTH * CELL_SIZE
HEIGHT = MAZE_HEIGHT * CELL_SIZE

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Maze Reinforcement Learning')

# Create maze using DFS algorithm
def create_maze():
    # Initialize maze grid with walls
    maze = np.ones((MAZE_HEIGHT * 2 + 1, MAZE_WIDTH * 2 + 1), dtype=int)
    
    # Create start and end points
    start = (1, 1)
    end = (MAZE_HEIGHT * 2 - 1, MAZE_WIDTH * 2 - 1)
    
    # Initialize visited cells
    visited = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
    
    # DFS stack
    stack = [(0, 0)]
    visited[0, 0] = True
    
    # DFS to create maze
    while stack:
        current_cell = stack[-1]
        x, y = current_cell
        
        # Convert to maze coordinates
        maze_x, maze_y = x * 2 + 1, y * 2 + 1
        maze[maze_y, maze_x] = 0  # Mark cell as path
        
        # Get unvisited neighbors
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT and not visited[ny, nx]:
                neighbors.append((nx, ny, dx, dy))
        
        if neighbors:
            # Choose a random neighbor
            nx, ny, dx, dy = random.choice(neighbors)

            
            # Remove wall between current and next
            maze[maze_y + dy, maze_x + dx] = 0
            
            # Mark next as visited and add to stack
            visited[ny, nx] = True
            stack.append((nx, ny))
        else:
            # Backtrack
            stack.pop()
    
    # Ensure start and end are open
    maze[start[0], start[1]] = 0
    maze[end[0], end[1]] = 0
    
    return maze, start, (MAZE_HEIGHT * 2 - 1, MAZE_WIDTH * 2 - 1)

# Q-learning agent
class QLearningAgent:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.height, self.width = maze.shape
        
        # Q-table: state -> action -> value
        self.q_table = {}
        
        # Initialize Q-table
        for y in range(self.height):
            for x in range(self.width):
                if maze[y, x] == 0:  # if it's a path
                    self.q_table[(y, x)] = {"up": 0, "down": 0, "left": 0, "right": 0}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
        # Current position
        self.current_pos = start
        
        # Movement mapping
        self.actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
    
    def reset(self):
        self.current_pos = self.start
        return self.current_pos
    
    def get_valid_actions(self, state):
        valid_actions = []
        y, x = state
        
        for action, (dy, dx) in self.actions.items():
            new_y, new_x = y + dy, x + dx
            
            # Check if new position is valid
            if (0 <= new_y < self.height and 
                0 <= new_x < self.width and 
                self.maze[new_y, new_x] == 0):  # if it's a path
                valid_actions.append(action)
        
        return valid_actions
    
    def choose_action(self, state):
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Choose random action from valid actions
            valid_actions = self.get_valid_actions(state)
            if valid_actions:
                return random.choice(valid_actions)
            return None
        else:
            # Choose best action based on Q-values
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                return None
            
            # Get Q-values for valid actions
            q_values = {action: self.q_table[state][action] for action in valid_actions}
            
            # Choose action with highest Q-value (randomly in case of ties)
            max_q = max(q_values.values())
            best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
            
            return random.choice(best_actions)
    
    def take_action(self, action):
        if action is None:
            return self.current_pos, -1, False
        
        # Get new position
        dy, dx = self.actions[action]
        new_y, new_x = self.current_pos[0] + dy, self.current_pos[1] + dx
        
        # Check if new position is valid
        if (0 <= new_y < self.height and 
            0 <= new_x < self.width and 
            self.maze[new_y, new_x] == 0):  # if it's a path
            
            # Update current position
            self.current_pos = (new_y, new_x)
            
            # Calculate reward
            if self.current_pos == self.goal:
                reward = 100  # High reward for reaching goal
                done = True
            else:
                reward = -1  # Small penalty for each step
                done = False
            
            return self.current_pos, reward, done
        
        return self.current_pos, -5, False  # Penalty for hitting a wall
    
    def update_q_table(self, state, action, reward, next_state):
        if action is None:
            return
        
        # Get best next action
        valid_actions = self.get_valid_actions(next_state)
        if valid_actions:
            max_next_q = max([self.q_table[next_state][a] for a in valid_actions])
        else:
            max_next_q = 0
        
        # Q-learning formula
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-value
        self.q_table[state][action] = new_q
    
    def update_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, 
                                    self.exploration_rate * self.exploration_decay)

# Main function
def main():
    # Create maze
    maze, start, goal = create_maze()
    
    # Initialize agent
    agent = QLearningAgent(maze, start, goal)
    
    # Set up clock
    clock = pygame.time.Clock()
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 200
    
    # Stats
    episode_rewards = []
    episode_steps = []
    
    # Training loop
    episode = 0
    running = True
    step_count = 0
    total_reward = 0
    state = agent.reset()
    done = False
    
    while running and episode < num_episodes:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Choose and take action
        action = agent.choose_action(state)
        next_state, reward, done = agent.take_action(action)
        
        # Print thinking process
        print(f"Episode: {episode+1}/{num_episodes}, Step: {step_count+1}")
        print(f"Position: {state}, Action: {action}, Reward: {reward}")
        print(f"Exploration rate: {agent.exploration_rate:.4f}")
        if action is not None:
            print(f"Q-values: {agent.q_table[state]}")
        print("-----")
        sys.stdout.flush()  # Ensure output is displayed immediately
        
        # Update Q-table
        agent.update_q_table(state, action, reward, next_state)
        
        # Update state and counters
        state = next_state
        step_count += 1
        total_reward += reward
        
        # Draw maze
        screen.fill(WHITE)
        cell_height, cell_width = maze.shape
        
        # Draw walls
        for y in range(cell_height):
            for x in range(cell_width):
                if maze[y, x] == 1:
                    pygame.draw.rect(screen, BLACK, 
                                    (x * CELL_SIZE / 2, y * CELL_SIZE / 2, 
                                     CELL_SIZE / 2, CELL_SIZE / 2))
        
        # Draw start and goal
        pygame.draw.rect(screen, GREEN, 
                         (start[1] * CELL_SIZE / 2, start[0] * CELL_SIZE / 2, 
                          CELL_SIZE / 2, CELL_SIZE / 2))
        pygame.draw.rect(screen, BLUE, 
                         (goal[1] * CELL_SIZE / 2, goal[0] * CELL_SIZE / 2, 
                          CELL_SIZE / 2, CELL_SIZE / 2))
        
        # Draw agent (X)
        pygame.draw.circle(screen, RED, 
                          (agent.current_pos[1] * CELL_SIZE / 2 + CELL_SIZE / 4, 
                           agent.current_pos[0] * CELL_SIZE / 2 + CELL_SIZE / 4), 
                          CELL_SIZE / 6)
        
        # Update display
        pygame.display.flip()
        
        # Episode end conditions
        if done or step_count >= max_steps_per_episode:
            # Log episode stats
            episode_rewards.append(total_reward)
            episode_steps.append(step_count)
            
            # Print episode summary
            print(f"\nEpisode {episode+1} completed!")
            print(f"Total steps: {step_count}")
            print(f"Total reward: {total_reward}")
            print(f"Final position: {agent.current_pos}")
            print("============\n")
            sys.stdout.flush()
            
            # Reset for next episode
            state = agent.reset()
            step_count = 0
            total_reward = 0
            done = False
            episode += 1
            
            # Update exploration rate
            agent.update_exploration_rate()
            
            # Small delay between episodes
            time.sleep(0.5)
        
        # Control speed
        clock.tick(10)  # Adjust for faster/slower visualization
    
    # Quit pygame
    pygame.quit()
    
    print("Training completed!")
    print(f"Final exploration rate: {agent.exploration_rate:.4f}")
    
    # Print final statistics
    if episode_rewards:
        print(f"Average reward per episode: {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(f"Average steps per episode: {sum(episode_steps) / len(episode_steps):.2f}")

if __name__ == "__main__":
    main()