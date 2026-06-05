"""Console entry points that register our tasks, then defer to mjlab's CLIs.

mjlab's own ``play``/``train`` only import mjlab's built-in tasks, so they can't
see ``Mjlab-PickPlace-Franka-*``. These thin wrappers import our task module
first (which registers into mjlab's shared registry) and then hand off to the
exact same mjlab entry points.
"""

from __future__ import annotations


def _register() -> None:
    import pick_place_challenge.tasks  # noqa: F401  (registration side effect)


def play() -> None:
    _register()
    from mjlab.scripts.play import main

    main()


def train() -> None:
    _register()
    from mjlab.scripts.train import main

    main()


def list_envs() -> None:
    """Print the registered Franka pick-and-place task IDs."""
    _register()
    import mjlab.tasks  # noqa: F401  (built-in tasks too)
    from mjlab.tasks.registry import list_tasks

    for task_id in list_tasks():
        marker = "*" if "Franka" in task_id else " "
        print(f"{marker} {task_id}")
