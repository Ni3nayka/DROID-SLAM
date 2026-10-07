from glob import glob
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPython(build_py):
    def run(self):
        super().run()
        # Bind a regular installation to this checkout too. A moved checkout can
        # override this with DROID_SLAM_ROOT; symlink builds discover it directly.
        root = next(p for p in Path(__file__).resolve().parents if (p / 'localization/map.py').is_file())
        target = Path(self.build_lib) / 'droid_slam' / '_build_location.py'
        target.write_text(f'REPOSITORY = {str(root)!r}\n')


setup(
    name='droid_slam', version='0.1.0', packages=['droid_slam'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/droid_slam']),
                ('share/droid_slam', ['package.xml', 'README.md']),
                ('share/droid_slam/launch', glob('launch/*.launch.py')),
                ('share/droid_slam/config', glob('config/*.yaml'))],
    install_requires=['setuptools'], zip_safe=False,
    description='ROS 2 adapter for the existing DROID map localizer',
    license='BSD-3-Clause', cmdclass={'build_py': BuildPython},
    entry_points={'console_scripts': [
        'localization = droid_slam.bootstrap:main',
        'video_publisher = droid_slam.bootstrap:video_main',
    ]},
)
