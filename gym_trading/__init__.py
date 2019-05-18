from gym.envs.registration import register
from gym_trading.envs.price_jump import PriceJump
from gym_trading.envs.market_maker import MarketMaker


register(
    id=PriceJump.id,
    entry_point='gym_trading.envs:PriceJump',
    max_episode_steps=1000000,
    nondeterministic=False
)

register(
    id=MarketMaker.id,
    entry_point='gym_trading.envs:MarketMaker',
    max_episode_steps=1000000,
    nondeterministic=False
)

print('Crypto-RL: registered = {}, {}.'.format(PriceJump.id, MarketMaker.id))
