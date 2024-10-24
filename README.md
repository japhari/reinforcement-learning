## Q-Learning - Reinforcement Learning
1. **Imports and Setup:**
   ```python
   import numpy as np
   import gymnasium as gym
   import random
   import time
   from IPython.display import clear_output
   import matplotlib.pyplot as plt
   ```
    - Imports necessary libraries: NumPy for array operations, Gym for creating the environment, `random` for random exploration, `time` for delay handling, and `matplotlib` for rendering visuals.

2. **Environment Initialization:**
   ```python
   env = gym.make('FrozenLake-v1', desc=None, render_mode="rgb_array", map_name="4x4", is_slippery=False)
   action_space_size = env.action_space.n
   state_space_size = env.observation_space.n
   q_table = np.zeros((state_space_size, action_space_size))
   ```
    - Sets up the *FrozenLake-v1* environment, a 4x4 grid where the agent learns to navigate to the goal.
    - The environment consists of discrete states and actions, and the `q_table` is initialized with zeros.

3. **Hyperparameters:**
   ```python
   num_episodes = 80000
   max_steps_per_episode = 100
   learning_rate = 0.8
   discount_rate = 0.99
   exploration_rate = 1
   max_exploration_rate = 1
   min_exploration_rate = 0.001
   exploration_decay_rate = 0.00005
   ```
    - Sets parameters for learning: the number of episodes, steps per episode, learning and discount rates, and exploration parameters for balancing exploration and exploitation.

4. **Main Q-Learning Algorithm:**
   ```python
   rewards_all_episodes = []
   for episode in range(num_episodes):
       state = env.reset()[0]
       done = False
       rewards_current_episode = 0
       for step in range(max_steps_per_episode):
           exploration_rate_threshold = random.uniform(0, 1)
           if exploration_rate_threshold > exploration_rate:
               action = np.argmax(q_table[state,:])
           else:
               action = env.action_space.sample()

           new_state, reward, done, truncated, info = env.step(action)

           q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

           state = new_state
           rewards_current_episode += reward
           if done:
               break

       print("Episode: " + str(episode) + " Reward of Current Episode: " + str(rewards_current_episode)  + " Exploration Rate: " + str(exploration_rate))
       exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
       rewards_all_episodes.append(rewards_current_episode)
   ```
    - This is the core loop for Q-Learning:
        - For each episode, the agent starts at a random state (`state = env.reset()[0]`).
        - Based on the exploration threshold, the agent either explores (takes random action) or exploits (uses the best action from `q_table`).
        - The Q-table is updated using the Bellman equation: the current Q-value is adjusted using the reward and the discounted max Q-value of the next state.
        - After each step, the agent moves to the new state (`state = new_state`), and the reward is accumulated.
        - The exploration rate decays over time to encourage more exploitation as the agent learns.

5. **Tracking Rewards and Exploration Rate:**
   ```python
   rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
   count = 1000

   print("********Average reward per thousand episodes********\\n")
   for r in rewards_per_thousand_episodes:
       print(count, ": ", str(sum(r/1000)))
       count += 1000

   print("\\n\\n********Q-table********\\n")
   print(q_table)
   ```
    - After training, the rewards are split into chunks of 1000 episodes, and the average reward per chunk is printed.
    - The final Q-table is printed to show the learned values for state-action pairs.

6. **Visualizing the Trained Agent:**
   ```python
   for episode in range(5):
       state = env.reset()[0]
       done = False
       print("*****EPISODE ", episode+1, "*****\\n\\n\\n\\n")
       time.sleep(1)
       for step in range(max_steps_per_episode):
           clear_output(wait=True)
           screen = env.render()
           plt.imshow(screen)
           plt.axis('off')
           plt.show()
           time.sleep(1)

           action = np.argmax(q_table[state,:])
           new_state, reward, done, truncated, info = env.step(action)
           if done:
               if reward == 1:
                   clear_output(wait=True)
                   screen = env.render()
                   plt.imshow(screen)
                   plt.axis('off')
                   plt.show()
                   time.sleep(3)
                   print("****You reached the goal!****")
               else:
                   print("****You fell through a hole!****")
                   time.sleep(3)
                   clear_output(wait=True)
               break
           state = new_state
   ```
    - Runs a few episodes after training and renders the environment to visualize the agent's performance.
    - It prints whether the agent reached the goal or fell into a hole.

7. **Placeholder for Q-Learning Algorithm:**
   ```python
   for episode in range(num_episodes):
       # initialize new episode params

       for step in range(max_steps_per_episode):
           # Exploration-exploitation trade-off
           # Take new action
           # Update Q-table
           # Set new state
           # Add new reward

       # Exploration rate decay
       # Add current episode reward to total rewards list
   ```
    - This seems to be a placeholder or incomplete section intended to repeat the Q-learning steps explained above.

This code trains an agent using Q-learning on the *FrozenLake* environment, decaying exploration over time, and updating the Q-values to maximize the cumulative reward over episodes.