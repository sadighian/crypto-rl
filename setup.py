from setuptools import setup

setup(
    name='mp-crypto-agent',
    version='1.0',
    packages=['gdax_connector',
              'bitfinex_connector',
              'common_components'],
    url='http://github.com/redbanies3ofthem/crypto',
    license='TBD',
    author='Jonathan',
    author_email='jonathan.m.sadighian@gmail.com',
    description='Connector to record crypto exchange level 3 market data',
    install_requires=['pymongo', 'requests',
                     'ujson', 'asyncio',
                     'sortedcontainers',
                     'numpy', 'websockets']
)
