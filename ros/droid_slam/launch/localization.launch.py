"""Launch the localizer; camera drivers remain independent ROS nodes."""
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('droid_slam'))
    defaults = yaml.safe_load((share / 'config/localization.yaml').read_text())['/**']['ros__parameters']

    def setup(context):
        config = LaunchConfiguration('params_file').perform(context)
        params = yaml.safe_load(Path(config).expanduser().read_text())['/**']['ros__parameters']
        for name, default in defaults.items():
            value = LaunchConfiguration(name).perform(context)
            if value != '':
                parsed = value if isinstance(default, str) else yaml.safe_load(value)
                if isinstance(default, float) and type(parsed) is int:
                    parsed = float(parsed)
                if type(parsed) is not type(default):
                    raise ValueError(f'Invalid type for launch argument {name}')
                params[name] = parsed
        return [Node(package='droid_slam', executable='localization', name='droid_slam',
                     parameters=[params], output='screen')]

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/localization.yaml')),
        *[DeclareLaunchArgument(name, default_value='', description=f'Override {name} from params_file') for name in defaults],
        OpaqueFunction(function=setup),
    ])
