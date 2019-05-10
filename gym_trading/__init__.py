from gym.envs.registration import register
from gym_trading.envs.trading_gym import TradingGym
from gym_trading.envs.market_maker import MarketMaker

register(
    id=TradingGym.id,
    entry_point='gym_trading.envs:TradingGym',
    max_episode_steps=1000000,
    nondeterministic=False
)

register(
    id=MarketMaker.id,
    entry_point='gym_trading.envs:MarketMaker',
    max_episode_steps=1000000,
    nondeterministic=False
)

print('Crypto-RL: registered = {}, {}.'.format(TradingGym.id, MarketMaker.id))
