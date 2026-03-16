from datetime import datetime
import gym
import numpy as np
import cv2
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from tensorboard_video_recorder import TensorboardVideoRecorder
from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv
from duckiebots_unreal_sim.holodeck_lane_following_env import UELaneFollowingEnv

class ImageWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.last_obs = None
        self._last_action = np.zeros(2, dtype=np.float32)

    def seed(self, seed=None):
        return []

    def reset(self):
        obs = self.env.reset()
        image = obs['image'].astype(np.uint8)
        self.last_obs = image
        self._last_action = np.zeros(2, dtype=np.float32)
        return image

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        image = obs['image'].astype(np.uint8)
        self.last_obs = image

        forward_vel = action[0]
        turning = action[1]     

        spin_penalty = abs(turning) * 0.3

        forward_bonus = max(0.0, forward_vel) * 0.1

        turn_jerk = abs(turning - self._last_action[1]) * 0.1

        shaped_rew = rew - spin_penalty + forward_bonus - turn_jerk
        self._last_action = action

        return image, float(shaped_rew), done, info

    def render(self, **kwargs):
        if self.last_obs is not None:
            return self.last_obs
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        self.env.close()



def make_duckiebot_env() -> UELaneFollowingEnv:
    env = UELaneFollowingEnv({
        "use_domain_randomization": True,
        "render_game_on_screen": False,
        "use_mask_observation": False,
        "return_rgb_and_mask_as_observation": False,
        "preprocess_rgb_observations_with_rcan": False,
        # "rcan_checkpoint_path": "/home/jblanier/Downloads/ckpt-91.onnx",
        "rcan_checkpoint_path": "/home/jblanier/Downloads/ckpt_9_nov_17.onnx",
        "launch_game_process": True,
        "simulate_latency": True,
        "normalize_image": False,
        "frame_stack_amount": 1,
        "physics_hz": 20,
        "use_simple_physics": False,
        "randomize_camera_location_for_tilted_robot": True,
        # "world_name": "DuckiebotsHolodeckMapDomainRandomization",
        "image_obs_only": False,
    })
    return ImageWrapper(env)



def main():
    experiment_name = "DuckieBotPPO_2M_n_envs=1_hyperparams_customrew_simple"
    experiment_log_dir = f"duckiebot_logs/{experiment_name}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    env = make_vec_env(make_duckiebot_env, n_envs=1)
    print(f'The duckiebot action space: {env.action_space}')
    print(f'The duckiebot observation space: {env.observation_space}')
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    video_trigger = lambda step: step % 20_000 == 0
    checkpoint = CheckpointCallback(save_freq=2000,
                                    save_path="./checkpoints/",
                                    name_prefix="duckiebot_ppo",
                                    verbose=1
                                   )
    env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=500,
                                   fps=30,
                                   tb_log_dir=experiment_log_dir)
    model = PPO(policy="CnnPolicy",
                env=env,
                verbose=1,
                tensorboard_log=experiment_log_dir,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                ent_coef=0.05,
               )
    try:
        model.learn(total_timesteps=2_000_000, callback=checkpoint)
        model.save("duckiebot_ppo_test_2M_1env_hyperparams_customrew_simple")
    except Exception as e:
        model.save("duckiebot_crash_save_2M_1env_hyperparams_customrew_simple")
    finally:
        model.save("duckiebot_ppo_test_2M_1envs_hyperparams_customrew_simple")
        env.close()

def resume_training(model_name: str, total_step: int, stop_step: int):
    experiment_name = "DuckieBotPPO_resume_500k_4env_custrew"
    experiment_log_dir = f"duckiebot_logs/{experiment_name}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    env = make_vec_env(make_duckiebot_env, n_envs=4)
    print(f'The duckiebot action space: {env.action_space}')
    print(f'The duckiebot observation space: {env.observation_space}')
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    video_trigger = lambda step: step % 10_000 == 0
    checkpoint = CheckpointCallback(save_freq=2000,
                                    save_path="./checkpoints/",
                                    name_prefix="duckiebot_ppo",
                                    verbose=1
                                   )
    env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=500,
                                   fps=30,
                                   tb_log_dir=experiment_log_dir)
    model = PPO.load(model_name, env=env)
    remain_step = total_step - stop_step
    model.learn(total_timesteps=remain_step, callback=checkpoint, reset_num_timesteps=False)

def test_model(model_name: str, n_episodes: int):
    experiment_name = "DuckieBotPPO_test_model_simple_rew"
    os.makedirs(f'videos/{experiment_name}')
    env = make_vec_env(make_duckiebot_env, n_envs=1)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    print("Loaded environment", flush=True)
    model = PPO.load(model_name, env=env)
    print("Loaded PPO model", flush=True)
    ep_rews = []
    ep_lens = []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_rew = 0.0
        ep_step = 0
        done = [False]
        video_path = f'videos/{experiment_name}/ep_{ep}.mp4'
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (64, 64))
        while not done[0] and ep_step < 500:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += reward[0]
            ep_step += 1
            frame = env.env_method("render")[0]
            frame_conv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_conv)
            print(f'Ep{ep} T{ep_step}: reward={reward[0]:.3f}', flush=True)
        writer.release()
        ep_rews.append(ep_rew)
        ep_lens.append(ep_step)
        print(f'Episode {ep + 1}: reward={ep_rew:.2f}, steps={ep_step}', flush=True)
    print(f'\n--- Results over {n_episodes} episodes ---', flush=True)
    print(f'Mean reward:  {np.mean(ep_rews):.2f}', flush=True)
    print(f'Std reward:   {np.std(ep_rews):.2f}', flush=True)
    print(f'Mean steps:   {np.mean(ep_lens):.1f}', flush=True)
    print(f'Best episode: {np.max(ep_lens):.2f}', flush=True)

    
    env.close()

if __name__ == "__main__":
    main()
    #resume_training("duckiebot_crash_save_500k_4env_hyperparams_customrew", 500_000, 455_640)
    #test_model("duckiebot_ppo_test_1M_1env_hyperparams_customrew_simple", 1)