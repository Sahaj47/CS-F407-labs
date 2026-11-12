import numpy as np
import matplotlib.pyplot as plt

class WarehouseEnvironment:
    def __init__(self):
        self.grid_size = 5
        self.start = (2, 0)
        self.goal = (2, 4)
        self.obstacles = {(0, 2), (1, 2), (3, 2), (4, 2)}
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Wait
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        self.max_steps = 40
        self.reset()
        
    def reset(self, start_pos=None):
        self.state = start_pos if start_pos else self.start
        self.steps = 0
        return self._get_state_idx(self.state)
        
    def _get_state_idx(self, state):
        # 25 states (5x5 grid)
        return state[0] * self.grid_size + state[1]
        
    def get_state_coords(self, idx):
        return (idx // self.grid_size, idx % self.grid_size)

    def step(self, action):
        self.steps += 1
        r, c = self.state
        dr, dc = self.actions[action]
        next_r, next_c = r + dr, c + dc
        
        # Check boundary and obstacles
        is_invalid = (next_r < 0 or next_r >= self.grid_size or 
                      next_c < 0 or next_c >= self.grid_size or 
                      (next_r, next_c) in self.obstacles)
        
        if is_invalid:
            next_state = self.state
            reward = -1.0
        else:
            next_state = (next_r, next_c)
            if next_state == self.goal:
                reward = 10.0
            else:
                reward = -0.1
                
        self.state = next_state
        terminated = (self.state == self.goal)
        truncated = (self.steps >= self.max_steps and not terminated)
        
        return self._get_state_idx(self.state), reward, terminated, truncated

def epsilon_greedy_selection(Q, state_idx, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, 5) # Explore
    else:
        # Exploit: break ties randomly
        q_values = Q[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return rng.choice(best_actions)

def q_learning_update(Q, s, a, r, next_s, terminated, alpha, gamma):
    # Bootstrap at truncation but not at goal termination
    if terminated:
        target = r
    else:
        target = r + gamma * np.max(Q[next_s])
        
    Q[s, a] += alpha * (target - Q[s, a])

def train_agent():
    env = WarehouseEnvironment()
    alpha = 0.2
    gamma = 0.95
    seed = 7
    num_episodes = 1500
    
    rng = np.random.default_rng(seed)
    Q = np.zeros((25, 5))
    
    returns = []
    successes = []
    
    for ep in range(num_episodes):
        s = env.reset()
        
        # Linearly decrease epsilon from 1 to 0.05 over 1,200 episodes
        if ep < 1200:
            epsilon = 1.0 - (1.0 - 0.05) * (ep / 1200.0)
        else:
            epsilon = 0.05
            
        ep_return = 0
        done = False
        
        while not done:
            a = epsilon_greedy_selection(Q, s, epsilon, rng)
            next_s, r, terminated, truncated = env.step(a)
            
            # Use next state from before reset
            q_learning_update(Q, s, a, r, next_s, terminated, alpha, gamma)
            
            ep_return += r
            s = next_s
            done = terminated or truncated
            
        returns.append(ep_return)
        successes.append(1 if terminated else 0)
        
    return Q, returns, successes

def evaluate_greedy(env, Q):
    s = env.reset()
    trajectory = [env.get_state_coords(s)]
    done = False
    
    while not done:
        a = np.argmax(Q[s]) # Fixed tie-breaking for evaluation
        next_s, r, terminated, truncated = env.step(a)
        trajectory.append(env.get_state_coords(next_s))
        s = next_s
        done = terminated or truncated
        
    return trajectory

def test_mode():
    print("--- Running Test Mode ---")
    env = WarehouseEnvironment()
    
    # Set custom position and check one action outcome
    start_pos = (2, 1)
    s = env.reset(start_pos=start_pos)
    print(f"Initial set position: {start_pos}")
    
    # Test action: Right (Action 3)
    action = 3 
    next_s, r, term, trunc = env.step(action)
    print(f"Action taken: {action} (Right)")
    print(f"Next position: {env.get_state_coords(next_s)}")
    print(f"Reward: {r}, Terminated: {term}, Truncated: {trunc}\n")

if __name__ == "__main__":
    # 1. Run isolated test mode
    test_mode()
    
    # 2. Train the agent
    print("--- Training Q-learning Agent ---")
    Q, returns, successes = train_agent()
    
    # 3. Evaluate greedy path
    env = WarehouseEnvironment()
    trajectory = evaluate_greedy(env, Q)
    print("Learned Greedy Trajectory:")
    print(" -> ".join([str(pos) for pos in trajectory]))
    
    # 4. Plot training curves (50-episode moving average)
    window = 50
    avg_returns = np.convolve(returns, np.ones(window)/window, mode='valid')
    avg_successes = np.convolve(successes, np.ones(window)/window, mode='valid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(avg_returns, color='blue')
    ax1.set_title(f'Training Returns (Moving Average, Window={window})')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Return')
    
    ax2.plot(avg_successes, color='green')
    ax2.set_title(f'Success Rate (Moving Average, Window={window})')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Success')
    
    plt.tight_layout()
    plt.show()