import os
from glob import glob

from setuptools import setup

package_name = 'ep_task3_pick'

setup(
    name=package_name,
    version='1.1.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'tools'),
            glob('tools/*.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuri',
    maintainer_email='yuri@example.com',
    description=(
        'Task3: RoboMaster EP 按 Grid 1..6 顺序视觉判别并抓取物体'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pick_node = ep_task3_pick.pick_node:main',
            'vision_node = ep_task3_pick.vision_node:main',
            'top_camera_node = ep_task3_pick.top_camera_node:main',
            'grid_calib = ep_task3_pick.grid_calib:main',
            'scene_randomizer = ep_task3_pick.scene_randomizer:main',
        ],
    },
)
