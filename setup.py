import os

from setuptools import setup

cwd = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(cwd, 'requirements.txt')
with open(file) as f:
    dependencies = list(map(lambda x: x.replace("\n", ""), f.readlines()))

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='crypto_rl',
      version='0.2.2',
      description='Cryptocurrency LOB trading environment in gym format.',
      long_description=long_description,
      author='Jonathan Sadighian',
      url='https://github.com/sadighian/crypto-rl',
      install_requires=dependencies,
      packages=['agent', 'data_recorder', 'gym_trading', 'indicators'])
