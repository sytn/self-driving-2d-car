import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from main import CircularRoadEnv
import time
import random
import pygame

# Enable training mode
Tensor.training = True

class PolicyNetwork:
    def __init__(self):
        # Network parameters
        self.l1 = Tensor.scaled_uniform(5, 32)  # input_size, hidden_size
        self.l2 = Tensor.scaled_uniform(32, 4)  # hidden_size, action_size
        self.opt = Adam([self.l1, self.l2], lr=0.001)
        
    def __call__(self, x: Tensor) -> Tensor:
        return x.dot(self.l1).tanh().dot(self.l2)
    
    def get_action(self, state: np.ndarray, epsilon=0.2) -> np.ndarray:
        if random.random() < epsilon:
            # Random exploration
            action = np.zeros(4)
            action[random.randint(0, 3)] = 1
            return action
        
        # Get action from policy
        with Tensor.no_grad():
            logits = self(Tensor(state.astype(np.float32))).numpy()
            action = np.zeros(4)
            action[np.argmax(logits)] = 1
            return action
    
    def train_step(self, states, actions, rewards):
        states_t = Tensor(np.array(states, dtype=np.float32))
        actions_t = Tensor(np.array(actions, dtype=np.float32))
        rewards_t = Tensor(np.array(rewards, dtype=np.float32))
        
        # Forward pass
        logits = self(states_t)
        probs = logits.softmax()
        
        # Policy gradient loss
        action_probs = (probs * actions_t).sum(axis=1)
        loss = -(action_probs.log() * rewards_t).mean()
        
        # Backward pass
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.numpy()

def train():
    env = CircularRoadEnv()
    policy = PolicyNetwork()
    episode = 0
    
    try:
        while True:
            state = env.reset()
            states, actions, rewards = [], [], []
            total_reward = 0
            done = False
            
            while not done:
                action = policy.get_action(state, epsilon=max(0.01, 0.5*(1 - episode/500)))
                next_state, reward, done = env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                state = next_state
                
                # Render
                env.render()
                time.sleep(0.02)
                
                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            
            # Discount rewards
            discounted = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(len(rewards))):
                running_add = running_add * 0.99 + rewards[t]
                discounted[t] = running_add
            
            # Normalize rewards
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-7)
            
            # Train
            if len(states) > 0:
                loss = policy.train_step(states, actions, discounted)
                print(f"Episode {episode}, Reward: {total_reward:.1f}, Loss: {loss:.4f}")
            
            episode += 1
            
    except KeyboardInterrupt:
        print("Training stopped by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train()