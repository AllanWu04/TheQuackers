from datetime import datetime
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from tensorboard_video_recorder import TensorboardVideoRecorder
from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv
from duckiebots_unreal_sim.holodeck_lane_following_env import UELaneFollowingEnv



class ImageWrapper(gym.Env):
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
        image = obs['image'].astype(np.uint8)
        self.last_obs = image
        return image

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        image = obs['image'].astype(np.uint8)
        self.last_obs = image
        return image, float(rew), done, info

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
        "image_obs_only": True
    })
    return ImageWrapper(env)



def main():
    experiment_name = "DuckieBotPPO_500k_n_envs=4"
    experiment_log_dir = f"duckiebot_logs/{experiment_name}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    env = make_vec_env(make_duckiebot_env, n_envs=4)
    print(f'The duckiebot action space: {env.action_space}')
    print(f'The duckiebot observation space: {env.observation_space}')
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    video_trigger = lambda step: step % 5000 == 0
    env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=500,
                                   fps=30,
                                   tb_log_dir=experiment_log_dir)
    model = PPO(policy="CnnPolicy",
                env=env,
                verbose=1,
                tensorboard_log=experiment_log_dir)
    model.learn(total_timesteps=500_000)
    model.save("duckiebot_ppo_test_500k_4envs")

def test_model(model_name: str, n_episodes: int):
    env = make_vec_env(make_duckiebot_env, n_envs=1)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load(model_name, env=env)
    print("Loaded PPO model")
    for ep in range(n_episodes):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

if __name__ == "__main__":
    main()
    test_model("duckiebot_ppo_test_500k_4envs", 200)