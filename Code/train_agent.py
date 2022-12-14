import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from RL_env import RL_regenwormen

#make directories
log_path = os.path.join("Logs")
PPO_load = os.path.join("Trained_Models", "RL_PL_3ms_Updated3")
PPO_save = os.path.join("Trained_Models", "RL_PL_3ms_Updated1")

env = DummyVecEnv([lambda: RL_regenwormen(render=False)])
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

model = PPO.load(PPO_load, env=env)
model.learn(total_timesteps=2000000)
model.save(PPO_save)

# TEST Model
'''episodes = 2
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs) #NOW USING MODEL HERE
        obs, reward, done, info = env.step(action)
        #env.render()
        score += reward
    print("Episode:{} Score:{}".format(episode, score))

env.reset()
#env.close()'''