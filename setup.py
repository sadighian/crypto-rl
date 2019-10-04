from setuptools import setup
import os

cwd = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(cwd, 'requirements.txt')
with open(file) as f:
    dependencies = list(map(lambda x: x.replace("\n", ""), f.readlines()))

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='crypto_rl',
      version='0.1.7',
      description='Cryptocurrency LOB trading environment in gym format.',
      long_description=long_description,
      author='Jonathan Sadighian',
      url='https://github.com/redbanies3ofthem/crypto-rl',
      install_requires=dependencies,
      packages=['agent', 'configurations', 'data_recorder', 'gym_trading', 'indicators'])
