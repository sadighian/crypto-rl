from setuptools import setup

setup(name='gym_trading',
      version='0.1.0',
      description='Trading environment in gym format',
      author='Jonathan Sadighian',
      url='https://github.com/redbanies3ofthem/crypto-rl',
      install_requires=['gym', 'arctic', 'sortedcontainers'],
      packages=['gym_trading']
      )
