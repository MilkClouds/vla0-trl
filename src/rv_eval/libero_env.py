"""LIBERO environment utilities for evaluation."""

import os
from typing import Dict, List, Optional, Tuple

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

ENV_RESOLUTION = 256


def get_evaluation_tasks(
    task_suite_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Get tasks to evaluate based on suite/task specification."""
    benchmark_dict = benchmark.get_benchmark_dict()
    tasks_to_evaluate = {}

    if task_suite_name is None and task_name is None:
        # All suites except libero_100
        for suite_name, suite_cls in benchmark_dict.items():
            if suite_name == "libero_100":
                continue
            if suite_name == "libero_90":
                continue  # TODO: do not skip this
            ts = suite_cls()
            tasks_to_evaluate[suite_name] = [t.name for t in ts.tasks]

    elif task_name is None:
        # All tasks in specified suite
        ts = benchmark_dict[task_suite_name]()
        tasks_to_evaluate[task_suite_name] = [t.name for t in ts.tasks]

    elif task_suite_name is None:
        # Find suite for specified task
        for suite_name, suite_cls in benchmark_dict.items():
            ts = suite_cls()
            for task in ts.tasks:
                if task.name == task_name:
                    tasks_to_evaluate[suite_name] = [task_name]
                    return tasks_to_evaluate

    else:
        tasks_to_evaluate[task_suite_name] = [task_name]

    return tasks_to_evaluate


def get_task_info(
    task_name: str,
    task_suite_name: Optional[str] = None,
) -> Tuple:
    """Get task, init states, max steps, and description."""
    benchmark_dict = benchmark.get_benchmark_dict()

    if task_suite_name is not None:
        ts = benchmark_dict[task_suite_name]()
        for i, task in enumerate(ts.tasks):
            if task.name == task_name:
                return (
                    task,
                    ts.get_task_init_states(i),
                    TASK_SUITE_MAX_STEPS.get(task_suite_name, 300),
                    task.language,
                )
    else:
        for suite_name, suite_cls in benchmark_dict.items():
            ts = suite_cls()
            for i, task in enumerate(ts.tasks):
                if task.name == task_name:
                    return (
                        task,
                        ts.get_task_init_states(i),
                        TASK_SUITE_MAX_STEPS.get(suite_name, 300),
                        task.language,
                    )

    raise ValueError(f"Task '{task_name}' not found")


def create_env(task, seed: int = 7) -> OffScreenRenderEnv:
    """Create LIBERO environment for a task."""
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
        camera_depths=True,
    )
    env.seed(seed)
    return env


def init_libero_env(
    task_name: str,
    task_suite_name: Optional[str] = None,
    seed: int = 7,
) -> Tuple:
    """Initialize LIBERO environment for evaluation.

    Returns:
        env: LIBERO environment
        init_states: List of initial states for evaluation
        max_steps: Maximum steps for the task
        instruction: Task instruction
    """
    task, init_states, max_steps, instruction = get_task_info(task_name, task_suite_name)
    env = create_env(task, seed)
    return env, init_states, max_steps, instruction


def get_observation(env, cam_list=("agentview_image", "robot0_eye_in_hand_image")):
    """Get observation from environment."""
    obs = env._get_observations()
    images = [obs[cam] for cam in cam_list]
    return images
