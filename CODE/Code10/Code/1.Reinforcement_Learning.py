# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:07:48 2023
@Created by: Noopa Jagadeesh
"""

## Step1. Import dependencies
import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os
from stable_baselines3.common.env_util import make_vec_env
#------------------------------------------------------------------------------
## Step2. Load Environment
environment_name = "CartPole-v0"
env = gym.make(environment_name,render_mode="human") #making our environment

#Understanding The Environment
#https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
episodes = 50
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info, _ = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

#env.reset()
#env.action_space.sample()
#env.observation_space.sample()

# 0-push cart to left, 1-push cart to the right
env.action_space.sample()

# [cart position, cart velocity, pole angle, pole angular velocity]
env.observation_space.sample()
#------------------------------------------------------------------------------
## Step3. Train an RL Model
log_path = os.path.join('Training','Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

#------------------------------------------------------------------------------
#Step4. Save and Reload Model
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model_CartPole')
model.save(PPO_path)

del model
model = PPO.load(PPO_path)

#------------------------------------------------------------------------------
#Step5. Evaluation
evaluate_policy(model, env, n_eval_episodes=10)
env.close()

#------------------------------------------------------------------------------
#Step6. Test Model
episodes = 50
env = make_vec_env(environment_name)
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        score += rewards
        env.render("human")
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
          






