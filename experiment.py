from agent.dqn import Agent
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('experiment')


parser = argparse.ArgumentParser()
parser.add_argument('--step_size',
                    default=1,
                    help="Increment for looping through historical data",
                    type=int)
parser.add_argument('--window_size',
                    default=1,
                    help="Number of lags to include in the observation",
                    type=int)
parser.add_argument('--max_position',
                    default=5,
                    help="Maximum number of positions that are " +
                         "able to be held in a broker's inventory",
                    type=int)
parser.add_argument('--fitting_file',
                    default='LTC-USD_2019-04-07.csv.xz',
                    help="Data set for fitting the z-score scaler (previous day)",
                    type=str)
parser.add_argument('--testing_file',
                    default='LTC-USD_2019-04-08.csv.xz',
                    help="Data set for training the agent (current day)",
                    type=str)
parser.add_argument('--id',
                    # default='market-maker-v0',
                    default='long-short-v0',
                    help="Environment ID; Either 'long-short-v0' or 'market-maker-v0'",
                    type=str)
parser.add_argument('--number_of_training_steps',
                    default=1e5,
                    help="Number of steps to train the agent "
                         "(does not include action repeats)",
                    type=int)
parser.add_argument('--gamma',
                    default=0.995,
                    help="Discount for future rewards",
                    type=float)
parser.add_argument('--seed',
                    default=1,
                    help="Random number seed for dataset",
                    type=int)
parser.add_argument('--action_repeats',
                    default=10,
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
parser.add_argument('--format_3d',
                    default=False,
                    help="Expand the observation space by one dimension" +
                    "to be compatible with CNNs. TRUE for Baselines, and FALSE for "
                    "Keras-RL." +
                    "E.g., [window, features] --> [window, features, 1]",
                    type=bool)
parser.add_argument('--z_score',
                    default=True,
                    help="If TRUE, normalize data with z-score",
                    type=bool)
parser.add_argument('--reward_type',
                    default='default',
                    choices=['trade_completion', 'continuous_total_pnl',
                             'continuous_realized_pnl', 'continuous_unrealized_pnl',
                             'normed', 'div', 'asymmetrical', 'asymmetrical_adj',
                             'default'],
                    help="""
                    reward_type: method for calculating the environment's reward:
                    1) 'trade_completion' --> reward is generated per trade's round trip.
                    2) 'continuous_total_pnl' --> change in realized & unrealized pnl  
                        between time steps.
                    3) 'continuous_realized_pnl' --> change in realized pnl between 
                        time steps.
                    4) 'continuous_unrealized_pnl' --> change in unrealized pnl 
                        between time steps.
                    5) 'normed' --> refer to https://arxiv.org/abs/1804.04216v1
                    6) 'div' --> reward is generated per trade's round trip divided by
                        inventory count (again, refer to 
                        https://arxiv.org/abs/1804.04216v1).
                    7) 'asymmetrical' --> 'default' enhanced with a reward for being  
                        filled above/below midpoint, and returns only negative rewards for 
                        Unrealized PnL to discourage long-term speculation.
                    8) 'asymmetrical_adj' --> 'default' enhanced with a reward for being  
                        filled above/below midpoint, and weighted up/down unrealized  
                        returns.
                    9) 'default' --> Pct change in Unrealized PnL + Realized PnL of  
                        respective time step/
                    """,
                    type=str)
parser.add_argument('--scale_rewards',
                    default=False,
                    help="If TRUE, scale PnL by a scalar defined in `broker.py`",
                    type=bool)
parser.add_argument('--nn_type',
                    default='mlp',
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
    logger.info('Experiment creating agent with kwargs: {}'.format(kwargs))
    agent = Agent(**kwargs)
    logger.info('Agent created. {}'.format(agent))
    agent.start()


if __name__ == '__main__':
    main(kwargs=args)
