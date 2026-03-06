from datetime import datetime
import gym
import numpy as np
import random

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from tensorboard_video_recorder import TensorboardVideoRecorder
from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv
from duckiebots_unreal_sim.holodeck_lane_following_env import UELaneFollowingEnv

#seeding for reproducing results
SEED = 0

def seed_all(seed):
    if seed is None:
       return
    random.seed(seed)
    np.random.seed(seed)
    try:
       import torch
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
    except Exception:
       pass



class ImageWrapper(gym.Env):
    """
    Converts the duckiebots dict observation to plain rgb image obseravations
    provides render() so tensorboard can grab frames
    """
    def __init__(self, env):
       super().__init__()
       self.env = env
       self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
       self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
       self.last_obs = None

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if hasattr(self.env, "seed"):
           try:
              self.env.seed(seed)
           except Exception:
              pass
        return [seed]

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
           self.seed(seed)
        obs = self.env.reset(**kwargs)
        image = obs["image"].astype(np.uint8)
        self.last_obs = image
        return image

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
    
        image = obs["image"].astype(np.uint8)
        self.last_obs = image

        return image, float(reward), done, info

    def render(self, **kwargs):
        if self.last_obs is not None:
           return self.last_obs
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        self.env.close()

def make_duckiebot_env():
    base_env = UELaneFollowingEnv({
    "use_domain_randomization": False,
    "render_game_on_screen": False,
    "use_mask_observation": False,
    "return_rgb_and_mask_as_observation": False,
    "preprocess_rgb_observations_with_rcan": False,
    "rcan_checkpoint_path": "/home/jblanier/Downloads/ckpt_9_nov_17.onnx",
    "launch_game_process": True,
    "simulate_latency": False,
    "normalize_image": False,
    "frame_stack_amount": 1,
    "physics_hz": 20,
    "use_simple_physics": False,
    "randomize_camera_location_for_tilted_robot": False,
    "image_obs_only": True
    })

    env = ImageWrapper(base_env)
    return env

def main():
    seed_all(SEED)
    experiment_name = "DuckieBotSAC_500k_hyperp3_2env_buf50k_SEED"
    log_dir = f"duckiebot_logs_SEED/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    env = make_vec_env(make_duckiebot_env, n_envs=2, seed=SEED)

    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    video_trigger = lambda step: step % 5000 == 0

    env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=400,
                                   fps=30,
                                   tb_log_dir=log_dir,
                                   tag="sac_rollout")

    obs = env.reset()
    print("obs checksum:", float(np.asarray(obs).sum()))
    print("sample action:", env.action_space.sample())
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    model = SAC(policy="CnnPolicy",
                env=env,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=50_000, #help with oom
                batch_size=128, #help with oom
                seed=SEED,
               #hyperparamters
                learning_rate=1e-4,
                learning_starts=10_000,
                gamma=0.98,
               )

    try:
       model.learn(total_timesteps=500_000)
       model.save("duckiebot_sac_500k_hyperp3_2env_buff50k_model_seed")
    finally:
       env.close()

if __name__ == "__main__":
    main()
