from setuptools import setup

setup(
    name='crypto',
    version='3',
    packages=['coinbase_connector', 'common_components', 'bitfinex_connector'],
    url='https://github.com/RedBanies3ofThem/crypto',
    license='',
    author='Jonathan Sadighian',
    author_email='jonathan.m.sadighian@gmail.com',
    description='Application to record streaming order and trade tick data from Coinbase and Bitfinex into an Arctic Tick Store',
    install_requires=['requests',
                        'asyncio',
                        'sortedcontainers',
                        'numpy',
                        'websockets',
                        'arctic',
                        'pytz',
                        'pandas',
                        'datetime']
)
