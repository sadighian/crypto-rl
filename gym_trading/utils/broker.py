# broker.py
#
#   Wrapper class implementing position.py to manage inventory and PnL
#
#
from configurations import LOGGER
from gym_trading.utils.order import LimitOrder, MarketOrder
from gym_trading.utils.position import Position


class Broker(object):
    pct_scale = 100.  # use as the denominator to scale PnL

    def __init__(self,
                 max_position: int = 1,
                 transaction_fee: bool = False):
        """
        Broker class is a wrapper for the PositionI class
        and is implemented in `gym_trading.py`

        :param max_position: (int) maximum number of positions agent can have open
            at a given time.
        :param transaction_fee: (bool) if TRUE, transaction fees are applied to
            executions, else No fees
        """
        self.transaction_fee = transaction_fee
        self.long_inventory = Position(side='long',
                                       max_position=max_position,
                                       transaction_fee=self.transaction_fee)
        self.short_inventory = Position(side='short',
                                        max_position=max_position,
                                        transaction_fee=self.transaction_fee)

    def __str__(self):
        return self.long_inventory.__str__() + "\n" + self.short_inventory.__str__()

    def reset(self) -> None:
        """
        Reset long and short inventories.

        :return: (void)
        """
        self.long_inventory.reset()
        self.short_inventory.reset()

    def add(self, order: MarketOrder or LimitOrder) -> bool:
        """
        Add / update an order.

        :param order: (Order) New order to be used for updating existing order
            or placing a new order
        :return: (bool) TRUE if order add action successfully completed,
            FALSE if already at position_max or unknown order.side
        """
        if order.side == 'long':
            is_added = self.long_inventory.add(order=order)
        elif order.side == 'short':
            is_added = self.short_inventory.add(order=order)
        else:
            raise ValueError('Broker.add() unknown order.side = %s' % order)
        return is_added

    def remove(self, order: MarketOrder or LimitOrder) -> float:
        """
        Remove position from inventory and return position PnL.

        :param order: (Order) Market order used to close position.
        :return: (bool) TRUE if position removed successfully
        """
        pnl = 0.
        if order.side == 'long':
            pnl += self.long_inventory.remove(netting_order=order)
        elif order.side == 'short':
            pnl += self.short_inventory.remove(netting_order=order)
        else:
            raise ValueError('Broker.remove() unknown order.side = %s' % order.side)
        return pnl

    def flatten_inventory(self, bid_price: float, ask_price: float) -> float:
        """
        Flatten all positions held in inventory.

        :param bid_price: (float) current bid price
        :param ask_price: (float) current ask price
        :return: (float) PnL from flattening inventory
        """
        LOGGER.debug('Flattening inventory. {} longs / {} shorts'.format(
            self.long_inventory_count, self.short_inventory_count
        ))
        long_pnl = self.long_inventory.flatten_inventory(price=bid_price)
        short_pnl = self.short_inventory.flatten_inventory(price=ask_price)
        return long_pnl + short_pnl

    def get_unrealized_pnl(self, bid_price: float, ask_price: float) -> float:
        """
        Unrealized PnL as a percentage gain.

        :return: (float) PnL %
        """
        long_pnl = self.long_inventory.get_unrealized_pnl(price=bid_price)
        short_pnl = self.short_inventory.get_unrealized_pnl(price=ask_price)
        return long_pnl + short_pnl

    @property
    def realized_pnl(self) -> float:
        """
        Realized PnL as a percentage gain.

        :return: (float) PnL %
        """
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl

    def get_total_pnl(self, bid_price: float, ask_price: float) -> float:
        """
        Unrealized + realized PnL.

        :param bid_price: (float) current bid price
        :param ask_price: (float) current ask price
        :return: (float) total PnL
        """
        return self.get_unrealized_pnl(bid_price, ask_price) + self.realized_pnl

    @property
    def long_inventory_count(self) -> int:
        """
        Number of long positions currently held in inventory.

        :return: (int) number of positions
        """
        return self.long_inventory.position_count

    @property
    def short_inventory_count(self) -> int:
        """
        Number of short positions currently held in inventory.

        :return: (int) number of positions
        """
        return self.short_inventory.position_count

    @property
    def total_trade_count(self) -> int:
        """
        Total number of long and short trades executed.

        :return: (int) number of executed trades
        """
        return self.long_inventory.total_trade_count + self.short_inventory.total_trade_count

    @property
    def net_inventory_count(self) -> int:
        """
        Net number of positions held in inventory (short or long).

        :return: (int) number of positions
        """
        return self.long_inventory_count - self.short_inventory_count

    @property
    def net_inventory_exposure(self) -> float:
        """
        Net exposure in notional terms.

        :return:
        """
        long_exposure = self.long_inventory.total_exposure
        short_exposure = self.short_inventory.total_exposure
        return long_exposure - short_exposure

    @property
    def total_inventory_notional(self) -> float:
        """
        Total exposure in [units * default size].

        :return:
        """
        return self.net_inventory_count * MarketOrder.DEFAULT_SIZE

    def step_limit_order_pnl(self, bid_price: float, ask_price: float, buy_volume: float,
                             sell_volume: float, step: int) -> (float, bool, bool):
        """
        Update PnL & positions every time step in the environment.

        :param bid_price: (float) current time step bid price
        :param ask_price: (float) current time step ask price
        :param buy_volume: (float) current time step buy volume
        :param sell_volume: (float) current time step sell volume
        :param step: (int) current time step number
        :return: (float) PnL for current time step due to limit order fill and netting
        """
        pnl = 0.
        is_long_order_filled = self.long_inventory.step(bid_price=bid_price,
                                                        ask_price=ask_price,
                                                        buy_volume=buy_volume,
                                                        sell_volume=sell_volume,
                                                        step=step)
        is_short_order_filled = self.short_inventory.step(bid_price=bid_price,
                                                          ask_price=ask_price,
                                                          buy_volume=buy_volume,
                                                          sell_volume=sell_volume,
                                                          step=step)

        if is_long_order_filled and is_short_order_filled:
            # protection in case Long and Short orders get filled in the same time step.
            # Although this shouldn't happen, it prevents an error from occurring if it
            # does happen.
            LOGGER.info("WARNING: Long and Short orders filled in the same step")
            LOGGER.info(
                'bid={} | ask={} | buy_vol={} | sell_vol={} | step={}'.format(
                    bid_price, ask_price, buy_volume, sell_volume, step)
            )
            is_short_order_filled = False

        if is_long_order_filled:
            # check if we can net the inventory
            if self.short_inventory_count > 0:
                # net out the inventory
                new_position = self.long_inventory.pop_position()
                pnl += self.short_inventory.remove(netting_order=new_position)

        if is_short_order_filled:
            # check if we can net the inventory
            if self.long_inventory_count > 0:
                # net out the inventory
                new_position = self.short_inventory.pop_position()
                pnl += self.long_inventory.remove(netting_order=new_position)

        return pnl, is_long_order_filled, is_short_order_filled

    def get_short_order_distance_to_midpoint(self, midpoint=100.) -> float:
        """
        Distance percentage between current midpoint price and the open order's
        posted price.

        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) distance between open order and midpoint price
        """
        return self.short_inventory.get_distance_to_midpoint(midpoint=midpoint)

    def get_long_order_distance_to_midpoint(self, midpoint=100.) -> float:
        """
        Distance percentage between current midpoint price and the open order's
        posted price.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) distance between open order and midpoint price
        """
        return self.long_inventory.get_distance_to_midpoint(midpoint=midpoint)

    def get_queues_ahead_features(self) -> tuple:
        """
        Scaled [-1, 1] ratio of how 'complete' an open order is. 'Complete' can be
        defined as a function with arguments:
            1. Open order's current rank in the LOB queue
            2. Partial fills in the open order
            3. Open order's quantity/size

        :return: (float) Scaled open order completion ratio [-1, 1]
        """
        buy_queue = short_queue = 0.

        if self.long_inventory.order:
            queue = self.long_inventory.order.queue_ahead
            executions = self.long_inventory.order.executed
            trade_size = self.long_inventory.order.DEFAULT_SIZE
            buy_queue = (executions - queue) / (queue + trade_size)

        if self.short_inventory.order:
            queue = self.short_inventory.order.queue_ahead
            executions = self.short_inventory.order.executed
            trade_size = self.short_inventory.order.DEFAULT_SIZE
            short_queue = (executions - queue) / (queue + trade_size)

        return buy_queue, short_queue

    @property
    def average_trade_pnl(self) -> float:
        """
        Average profit (or loss) per trade, given the trade history.

        :return: (float) average pnl per trade
        """
        if self.realized_pnl == 0.:
            return 0.
        elif self.total_trade_count == 0.:
            return 0.
        return self.realized_pnl / self.total_trade_count

    def get_statistics(self) -> dict:
        """
        Get statistics for long and short inventories.

        :return: statistics
        """
        return dict(
            short_inventory_market_orders=self.short_inventory.statistics.market_orders,
            short_inventory_orders_placed=self.short_inventory.statistics.orders_placed,
            short_inventory_orders_updated=self.short_inventory.statistics.orders_updated,
            short_inventory_orders_executed=self.short_inventory.statistics.orders_executed,
            long_inventory_market_orders=self.long_inventory.statistics.market_orders,
            long_inventory_orders_placed=self.long_inventory.statistics.orders_placed,
            long_inventory_orders_updated=self.long_inventory.statistics.orders_updated,
            long_inventory_orders_executed=self.long_inventory.statistics.orders_executed
            )
