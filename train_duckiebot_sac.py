from datetime import datetime
import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from tensorboard_video_recorder import TensorboardVideoRecorder
from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv
from duckiebots_unreal_sim.holodeck_lane_following_env import UELaneFollowingEnv

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
        return []

    def reset(self):
        obs = self.env.reset()
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
    "use_domain_randomization": True,
    "render_game_on_screen": False,
    "use_mask_observation": False,
    "return_rgb_and_mask_as_observation": False,
    "preprocess_rgb_observations_with_rcan": False,
    "rcan_checkpoint_path": "/home/jblanier/Downloads/ckpt_9_nov_17.onnx",
    "launch_game_process": True,
    "simulate_latency": True,
    "normalize_image": False,
    "frame_stack_amount": 1,
    "physics_hz": 20,
    "use_simple_physics": False,
    "randomize_camera_location_for_tilted_robot": True,
    })

    return ImageWrapper(base_env)

def main():
    experiment_name = "DuckieBotSAC"
    log_dir = f"duckiebot_logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    env = make_vec_env(make_duckiebot_env, n_envs=1)

    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    video_trigger = lambda step: step % 1000 == 0

    env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=400,
                                   fps=30,
                                   tb_log_dir=log_dir,
                                   tag="sac_rollout")

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    model = SAC(policy="CnnPolicy",
                env=env,
                verbose=1,
                tensorboard_log=log_dir,
                
               #hyperparamters
               buffer_size=200_000,
               learning_starts=10_000,
               batch_size=256,
               train_freq=1,
               gradient_steps=1,
               gamma=0.99,
               tau=0.005,
               ent_coef="auto",
               learning_rate=3e-4)

    model.learn(total_timesteps=100_000)
    model.save("duckiebot_sac_model")

if __name__ == "__main__":
    main()
