import argparse

from agent.dqn import Agent
from configurations import LOGGER

parser = argparse.ArgumentParser()
parser.add_argument('--window_size',
                    default=100,
                    help="Number of lags to include in the observation",
                    type=int)
parser.add_argument('--max_position',
                    default=5,
                    help="Maximum number of positions that are " +
                         "able to be held in a broker's inventory",
                    type=int)
parser.add_argument('--fitting_file',
                    default='demo_LTC-USD_20190926.csv.xz',
                    help="Data set for fitting the z-score scaler (previous day)",
                    type=str)
parser.add_argument('--testing_file',
                    default='demo_LTC-USD_20190926.csv.xz',
                    help="Data set for training the agent (current day)",
                    type=str)
parser.add_argument('--symbol',
                    default='LTC-USD',
                    help="Name of currency pair or instrument",
                    type=str)
parser.add_argument('--id',
                    # default='market-maker-v0',
                    default='trend-following-v0',
                    help="Environment ID; Either 'trend-following-v0' or "
                         "'market-maker-v0'",
                    type=str)
parser.add_argument('--number_of_training_steps',
                    default=1e5,
                    help="Number of steps to train the agent "
                         "(does not include action repeats)",
                    type=int)
parser.add_argument('--gamma',
                    default=0.99,
                    help="Discount for future rewards",
                    type=float)
parser.add_argument('--seed',
                    default=1,
                    help="Random number seed for data set",
                    type=int)
parser.add_argument('--action_repeats',
                    default=5,
                    help="Number of steps to pass on between actions",
                    type=int)
parser.add_argument('--load_weights',
                    default=False,
                    help="Load saved load_weights if TRUE, otherwise start from scratch",
                    type=bool)
parser.add_argument('--visualize',
                    default=False,
                    help="Render midpoint on a screen",
                    type=bool)
parser.add_argument('--training',
                    default=True,
                    help="Training or testing mode. " +
                         "If TRUE, then agent starts learning, " +
                         "If FALSE, then agent is tested",
                    type=bool)
parser.add_argument('--reward_type',
                    default='default',
                    choices=['default',
                             'default_with_fills',
                             'realized_pnl',
                             'differential_sharpe_ratio',
                             'asymmetrical',
                             'trade_completion'],
                    help="""
                    reward_type: method for calculating the environment's reward:
                    1) 'default' --> inventory count * change in midpoint price returns
                    2) 'default_with_fills' --> inventory count * change in midpoint  
                    price returns + closed trade PnL
                    3) 'realized_pnl' --> change in realized pnl between time steps
                    4) 'differential_sharpe_ratio' -->
                    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1
                    &type=pdf
                    5) 'asymmetrical' --> extended version of *default* and enhanced 
                    with  a reward for being filled above or below midpoint, 
                    and returns only negative rewards for Unrealized PnL to discourage 
                    long-term speculation.
                    6) 'trade_completion' --> reward is generated per trade's round trip
                    """,
                    type=str)
parser.add_argument('--nn_type',
                    default='cnn',
                    help="Type of neural network to use: 'cnn' or 'mlp' ",
                    type=str)
parser.add_argument('--dueling_network',
                    default=True,
                    help="If TRUE, use Dueling architecture in DQN",
                    type=bool)
parser.add_argument('--double_dqn',
                    default=True,
                    help="If TRUE, use double DQN for Q-value estimation",
                    type=bool)
args = vars(parser.parse_args())


def main(kwargs):
    LOGGER.info(f'Experiment creating agent with kwargs: {kwargs}')
    agent = Agent(**kwargs)
    LOGGER.info(f'Agent created. {agent}')
    agent.start()


if __name__ == '__main__':
    main(kwargs=args)
