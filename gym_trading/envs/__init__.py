from gym_trading.envs.market_maker import MarketMaker
from gym_trading.envs.trend_following import TrendFollowing


def test_env_loop(env) -> bool:
    """
    Evaluate a RL agent
    """
    total_reward = 0.0
    reward_list = []
    actions_tracker = dict()

    i = 0
    done = False
    env.reset()
    while not done:
        i += 1

        action = env.action_space.sample()

        state, reward, done, _ = env.step(action)
        total_reward += reward
        reward_list.append(reward)

        if action in actions_tracker:
            actions_tracker[action] += 1
        else:
            actions_tracker[action] = 1

        if done:
            print(f"Max reward: {max(reward_list)}\nMin reward: {min(reward_list)}")
            print(f"Agent completed {env.broker.total_trade_count} trades")
            # Visualize results
            env.plot_observation_history()
            env.plot_trade_history()
            break

    print(f"Total reward: {total_reward}")
    for action, count in actions_tracker.items():
        print(f"Action #{action} =\t{count}")

    return done
