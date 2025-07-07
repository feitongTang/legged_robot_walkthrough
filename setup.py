from setuptools import find_packages
from distutils.core import setup

setup(name='legged_robot_walkthrough',
      version='1.0.0',
      author='Tang, Tommy Feitong',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='feitong_Tang@163.com',
      description='Walkthrough for legged robot',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.24', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
