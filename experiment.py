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
                    default=50,
                    help="Number of lags to include in the observation",
                    type=int)
parser.add_argument('--train',
                    default=True,
                    help="Training or testing mode. " +
                    "If True, then agent starts learning, " +
                    "If False, then agent is tested",
                    type=bool)
parser.add_argument('--max_position',
                    default=1,
                    help="Maximum number of positions that are " +
                         "able to be held in a broker's inventory",
                    type=int)
parser.add_argument('--weights',
                    default=False,
                    help="Load saved weights if True, otherwise start from scratch",
                    type=bool)
parser.add_argument('--fitting_file',
                    default='ETH-USD_2018-12-31.xz',
                    help="Dataset for fitting the z-score scaler (previous day)",
                    type=str)
parser.add_argument('--testing_file',
                    default='ETH-USD_2019-01-01.xz',
                    help="Dataset for training the agent (current day)",
                    type=str)
parser.add_argument('--seed',
                    default=1,
                    help="Random number seed for environment",
                    type=int)
parser.add_argument('--frame_stack',
                    default=False,
                    help="Stack 4 snapshots as one observation if True, " +
                         "Otherwise 1 snapshot is the observation",
                    type=bool)
args = vars(parser.parse_args())


def main(kwargs):
    logger.info('Experiment creating agent with kwargs: {}'.format(kwargs))
    agent = Agent(**kwargs)
    logger.info('Agent created. {}'.format(agent))
    agent.start()
    logger.info('Training started.')


if __name__ == '__main__':
    main(kwargs=args)
