import pathlib

import cv2
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Int

from mtrl.types import Agent


def _write_video(
    frames: Int[npt.NDArray, "time height width 3"], filename: pathlib.Path, fps: int
) -> None:
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


def record_videos(
    envs: gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv,
    task_names: list[str],
    agent: Agent,
    output_dir: pathlib.Path,
    max_episodes: int = 50,
) -> None:
    """Records a video of the agent performing tasks.
    The vector dimension in the vector env is assumed to correspond to a unique task we want a video for.
    Records the first success and first failure for each task, or continues until max_episodes if either
    isn't encountered.

    Args:
        envs: The vector environment to record videos for.
        task_names: List of task names corresponding to each env in the vector env.
        agent: The agent to use for recording videos.
        output_dir: Directory to save the videos in.
        max_episodes: Maximum number of episodes to try per task.
    """
    assert isinstance(envs, gym.vector.AsyncVectorEnv) or isinstance(
        envs, gym.vector.SyncVectorEnv
    )
    assert (
        len(task_names) == envs.num_envs
    ), "Must provide a task name for each environment"

    render_fps = envs.metadata["render_fps"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track episodes and recordings for each env
    completed_episodes = np.zeros(envs.num_envs)
    success_recorded = np.zeros(envs.num_envs, dtype=bool)
    failure_recorded = np.zeros(envs.num_envs, dtype=bool)

    # Initialize frame buffers for each env
    current_frames = [[] for _ in range(envs.num_envs)]

    obs, _ = envs.reset()

    while not np.all(
        (success_recorded & failure_recorded) | (completed_episodes >= max_episodes)
    ):
        # Record frames for all active environments
        renders = envs.render()
        assert renders is not None
        for env_idx in range(envs.num_envs):
            if not (success_recorded[env_idx] and failure_recorded[env_idx]):
                current_frames[env_idx].append(renders[env_idx])

        # Get actions from agent
        actions = agent.eval_action(obs)

        # Step environments
        obs, _, terminated, truncated, infos = envs.step(actions)

        # Check for episode completion and handle recordings
        for env_idx in range(envs.num_envs):
            if terminated[env_idx] or truncated[env_idx]:
                completed_episodes[env_idx] += 1

                # Check for success
                if (
                    terminated[env_idx]
                    and bool(infos["success"][env_idx])
                    and not success_recorded[env_idx]
                ):
                    success_path = output_dir / f"{task_names[env_idx]}_success.mp4"
                    _write_video(
                        np.array(current_frames[env_idx]), success_path, render_fps
                    )
                    success_recorded[env_idx] = True

                # Check for failure
                if (
                    truncated[env_idx]
                    and not bool(infos["success"][env_idx])
                    and not failure_recorded[env_idx]
                ):
                    failure_path = output_dir / f"{task_names[env_idx]}_failure.mp4"
                    _write_video(
                        np.array(current_frames[env_idx]), failure_path, render_fps
                    )
                    failure_recorded[env_idx] = True

                # Reset frame buffer for this env
                current_frames[env_idx] = []
