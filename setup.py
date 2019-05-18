from setuptools import setup
import os


cwd = os.path.dirname(os.path.realpath(__file__))
with open('{}/requirements.txt'.format(cwd)) as f:
      dependencies = list(map(lambda x: x.replace("\n", ""),
                              f.readlines()))

setup(name='gym_trading',
      version='0.1.2',
      description='Trading environment in gym format',
      author='Jonathan Sadighian',
      url='https://github.com/redbanies3ofthem/crypto-rl',
      install_requires=dependencies,
      packages=['gym_trading']
      )
