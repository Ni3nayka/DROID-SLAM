"""Select the project's tested Python environment before importing native modules."""

import os
from pathlib import Path
import sys


def repository():
    explicit = os.environ.get('DROID_SLAM_ROOT')
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = list(Path(__file__).resolve().parents)
        try:
            from ._build_location import REPOSITORY
            candidates.append(Path(REPOSITORY))
        except ImportError:
            pass
    for root in candidates:
        if (root / 'localization/realtime.py').is_file():
            return root.resolve()
    raise RuntimeError('Cannot locate DROID-SLAM checkout. Set DROID_SLAM_ROOT to its absolute path.')


def prepare():
    root = repository()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _launch(module):
    root = prepare()
    python = Path(os.environ.get('DROID_SLAM_PYTHON', str(root / '.venv/bin/python'))).expanduser()
    if not python.is_file():
        raise RuntimeError(f'Missing project Python: {python}. Set DROID_SLAM_PYTHON if using another environment.')
    # Preserve ROS paths while making both the adapter and core importable by
    # spawned matcher/GUI processes. Do not depend on the shell's working dir.
    env = os.environ.copy()
    env['DROID_SLAM_ROOT'] = str(root)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parent.parent), str(root),
                                       env.get('PYTHONPATH', '')])
    os.execve(str(python), [str(python), '-m', module, *sys.argv[1:]], env)


def main():
    _launch('droid_slam.node')


def video_main():
    _launch('droid_slam.video_publisher')
