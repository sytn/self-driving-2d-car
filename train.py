import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from main import CircularRoadEnv
import time
import random
import pygame
import math

Tensor.training = True

class PolicyNetwork:
    def __init__(self):
        # Network parameters
        self.l1 = Tensor.scaled_uniform(7, 64, a=-0.5, b=0.5)
        self.l2 = Tensor.scaled_uniform(64, 32, a=-0.5, b=0.5)
        self.l3 = Tensor.scaled_uniform(32, 4, a=-0.5, b=0.5)
        self.opt = Adam([self.l1, self.l2, self.l3], lr=0.001)
        
    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.l1).tanh().dot(self.l2).tanh().dot(self.l3)
    
    def get_action(self, state: np.ndarray, episode: int) -> np.ndarray:
        epsilon = max(0.01, 0.5 * (0.99 ** episode))
        
        if random.random() < epsilon:
            action_idx = random.randint(0, 3)
            return np.array([1 if i == action_idx else 0 for i in range(4)], dtype=np.float32)
        
        # Temporarily disable training mode to avoid gradient computation
        original_training = Tensor.training
        Tensor.training = False
        try:
            logits = self.forward(Tensor(state.astype(np.float32))).numpy()
            action = np.zeros(4)
            action[np.argmax(logits)] = 1
        finally:
            Tensor.training = original_training  # Restore original training state
        
        return action
    
    def train_step(self, states, actions, rewards):
        states_t = Tensor(np.array(states, dtype=np.float32))
        actions_t = Tensor(np.array(actions, dtype=np.float32))
        rewards_t = Tensor(np.array(rewards, dtype=np.float32))
        
        logits = self.forward(states_t)
        probs = logits.softmax()
        
        action_probs = (probs * actions_t).sum(axis=1)
        loss = -(action_probs.log() * rewards_t).mean()
        
        self.opt.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for t in [self.l1, self.l2, self.l3]:
            t.grad = t.grad.clip(-1, 1)
            
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
                action = policy.get_action(state, episode)
                next_state, reward, done = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                state = next_state
                
                env.render()
                pygame.time.delay(20)
                
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
            mean = discounted.mean()
            std = discounted.std()
            discounted = (discounted - mean) / (std + 1e-7)
            
            if len(states) > 10:  # Only train on meaningful episodes
                loss = policy.train_step(states, actions, discounted)
                print(f"Ep {episode:4d} | R: {total_reward:7.1f} | L: {loss:7.4f} | ε: {max(0.01, 0.5*(0.99**episode)):.3f}")
            
            episode += 1
            
    except KeyboardInterrupt:
        print("Training stopped by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train()