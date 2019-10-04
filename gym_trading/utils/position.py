# position.py
#
#   Inventory and risk management module for environments
#
#
import logging
from configurations.configs import MARKET_ORDER_FEE
from gym_trading.utils.order import Order

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('broker')


class Position(object):
    slippage = MARKET_ORDER_FEE / 4

    def __init__(self, side='long', max_position=1, transaction_fee=None):
        """
        Position class keeps track the agent's trades and provides stats (e.g.,
        pnl) on all trades.
        :param side: 'long' or 'short'
        :param max_position: (int) maximum number of positions agent can have open
                                    at a given time.
        :param transaction_fee: (float) fee to use for add/remove order transactions;
                If NONE, then transaction fees are omitted.
        """
        self.max_position_count = max_position
        self.positions = []
        self.realized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.side = side
        self.average_price = 0.0
        self.total_trade_count = 0
        self.transaction_fee = transaction_fee
        self.order = None

    def __str__(self):
        msg = 'PositionI-{}: [realized_pnl={:.4f}'.format(self.side, self.realized_pnl)
        msg += ' | total_exposure={:.4f} | total_trade_count={}]'.format(
            self.total_exposure, self.total_trade_count)
        return msg

    def reset(self):
        """
        Reset broker metrics / inventories.
        :return: (void)
        """
        self.positions.clear()
        self.realized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.average_price = 0.0
        self.total_trade_count = 0
        self.order = None

    @property
    def position_count(self):
        """
        Number of positions held in inventory.
        :return: (int) number of positions in inventory
        """
        return len(self.positions)

    def _process_transaction_volume(self, volume: float):
        """
        Handle order executions in LOB queue.
        :param volume: unsigned notional value of trades (either buy or sell)
        :return:(void)
        """
        if self.order.is_first_in_queue:
            self.order.process_executions(volume)
        else:
            self.order.reduce_queue_ahead(volume)

    def _step_limit_order(self, bid_price: float, ask_price: float, buy_volume: float,
                          sell_volume: float, step: int):
        """
        Step in environment and update LIMIT order inventories.
        :param bid_price: best bid price
        :param ask_price: best ask price
        :param buy_volume: executions initiated by buyers (in notional terms)
        :param sell_volume: executions initiated by sellers (in notional terms)
        :param step: current time step
        :return: (bool) TRUE if a limit order was filled, otherwise FALSE
        """
        if self.order is None:
            return False

        if self.order.side == 'long':
            if bid_price <= self.order.price:
                self._process_transaction_volume(volume=sell_volume)

        elif self.order.side == 'short':
            if ask_price >= self.order.price:
                self._process_transaction_volume(volume=buy_volume)

        if self.order.is_filled:
            avg_execution_px = self.order.get_average_execution_price()
            self.positions.append(self.order)
            self.total_exposure += avg_execution_px
            self.average_price = self.total_exposure / self.position_count
            self.full_inventory = self.position_count >= self.max_position_count
            self.total_trade_count += 1
            logger.debug('FILLED {} order #{} at {:.3f} after {} steps on {}.'.format(
                self.order.side, self.order.id, avg_execution_px,
                self.order.metrics.steps_in_position, step))
            self.order = None  # set the slot back to no open orders
            return True

        return False

    def _step_position_metrics(self, bid_price: float, ask_price: float, step: int):
        """
        Step in environment and update position metrics.
        :param bid_price: best bid price
        :param ask_price: best ask price
        :param step: current time step
        :return: (void)
        """
        price = bid_price if self.side == 'long' else ask_price
        if self.order:
            self.order.update_metrics(price=price, step=step)
        if self.position_count > 0:
            for position in self.positions:
                position.update_metrics(price=price, step=step)

    def step(self, bid_price: float, ask_price: float, buy_volume: float,
             sell_volume: float, step: int):
        """
        Step in environment and update broker inventories.
        :param bid_price: best bid price
        :param ask_price: best ask price
        :param buy_volume: executions initiated by buyers (in notional terms)
        :param sell_volume: executions initiated by sellers (in notional terms)
        :param step: current time step
        :return: (bool) TRUE if a limit order was filled, otherwise FALSE
        """
        is_filled = self._step_limit_order(bid_price=bid_price, ask_price=ask_price,
                                           buy_volume=buy_volume, sell_volume=sell_volume,
                                           step=step)
        self._step_position_metrics(bid_price=bid_price, ask_price=ask_price, step=step)
        return is_filled

    def cancel_limit_order(self):
        """
        Cancel a limit order
        :return: (bool) TRUE if cancel was successful
        """
        if self.order is None:
            logger.debug('No {} open orders to cancel.'.format(self.side))
            return False

        logger.debug('Cancelling order ({})'.format(self.order))
        self.order = None
        return True

    def _add_market_order(self, order: Order):
        """
        Add a MARKET order
        :param order: (Order) New order to be used for updating existing order or
                        placing a new order
        """
        if self.full_inventory:
            logger.debug('  %s inventory max' % order.side)
            return False

        # Create a hypothetical average execution price incorporating a fixed slippage
        hypothetical_avg_price = order.price + Position.slippage \
            if order.side == 'long' else order.price - Position.slippage
        order.average_execution_price = hypothetical_avg_price
        order.executed = Order.DEFAULT_SIZE

        # Update position inventory attributes
        self.cancel_limit_order()  # remove any unfilled limit orders
        self.positions.append(order)  # execute and save the market order
        self.total_exposure += order.average_execution_price
        self.average_price = self.total_exposure / self.position_count
        self.full_inventory = self.position_count >= self.max_position_count
        self.total_trade_count += 1
        logger.debug('  %s @ %.2f | step %i' % (
            order.side, order.average_execution_price, order.step))
        return True

    def _add_limit_order(self, order: Order):
        """
        Add / update a LIMIT order
        :param order: (Order) New order to be used for updating existing order or
                        placing a new order
        """
        if self.order is None:
            if self.full_inventory:
                logger.debug(("{} order rejected. Already at max position limit "
                             "({})").format(self.side, self.max_position_count))
                return False
            self.order = order
            logger.debug('\nOpened new order={}'.format(order))

        elif self.order.price != order.price:
            self.order.price = order.price
            self.order.queue_ahead = order.queue_ahead
            self.order.id = order.id
            self.order.step = order.step
            logger.debug('\nUpdating order{} --> \n{}'.format(order, self.order))

        else:
            logger.debug("\nNothing to update about the order {}".format(self.order))

        return True

    def add(self, order: Order):
        """
        Add / update an order
        :param order: (Order) New order to be used for updating existing order or
                        placing a new order
        :return: (bool) TRUE if order add action successfully completed, FALSE if already
                        at position_max or unknown order.side
        """
        if order.order_type == 'market':
            return self._add_market_order(order=order)
        elif order.order_type == 'limit':
            return self._add_limit_order(order=order)
        else:
            logger.warning(
                "Position() add --> unknown order_type {}".format(order.order_type))

    def remove(self, price: float):
        """
        Remove position from inventory and return position PnL.
        :param price: (float) Price to use for the PnL calculation
        :return: (bool) TRUE if position removed successfully
        """
        pnl = 0.
        if self.position_count < 1:
            logger.info('Error. No {} positions to remove.'.format(self.side))
            return pnl

        order = self.positions.pop(0)

        # Calculate PnL
        if self.side == 'long':
            pnl = (price - order.average_execution_price) / order.average_execution_price
        elif self.side == 'short':
            pnl = (order.average_execution_price - price) / order.average_execution_price

        # Add transaction fees, if provided
        if self.transaction_fee:
            pnl -= self.transaction_fee * 2  # round trip fees for position

        # Add Profit and Loss to realized gains/losses
        self.realized_pnl += pnl

        # Update positions attributes
        self.total_exposure -= order.average_execution_price
        self.average_price = self.total_exposure / self.position_count if \
            self.position_count > 0 else 0.
        self.full_inventory = self.position_count >= self.max_position_count
        logger.debug(
            'Netted {} position #{} with PnL = {:.4f}'.format(self.side, order.id, pnl))
        return pnl

    def pop_position(self):
        """
        Remove LIMIT order position from inventory when netted out.
        :return: (LimitOrder) position being netted out
        """
        if self.position_count > 0:
            position = self.positions.pop(0)

            # update positions attributes
            self.total_exposure -= position.average_execution_price
            if self.position_count > 0:
                self.average_price = self.total_exposure / self.position_count
            else:
                self.average_price = 0

            self.full_inventory = self.position_count >= self.max_position_count
            logger.debug('---%s position #%i @ %.4f has been netted out.' % (
                self.side, position.id, position.price))
            return position
        else:
            logger.info('Error. No {} pop_position to remove.'.format(self.side))
            return None

    def get_unrealized_pnl(self, price: float):
        """
        Unrealized PnL as a percentage gain
        :return: (float) PnL %
        """
        if self.position_count == 0:
            return 0.0

        if self.side == 'long':
            unrealized_pnl = (price - self.average_price) / self.average_price
        elif self.side == 'short':
            unrealized_pnl = (self.average_price - price) / self.average_price
        else:
            unrealized_pnl = 0.0
            logger.info(('Error: PositionI.get_unrealized_pnl() for '
                         'side = {}').format(self.side))

        # Add transaction fees, if provided
        if self.transaction_fee:
            unrealized_pnl -= self.transaction_fee * 2  # include round trip fees

        return unrealized_pnl

    def flatten_inventory(self, price: float):
        """
        Flatten all positions held in inventory.
        :param price: (float) current bid or ask price
        :return: (float) PnL from flattening inventory
        """
        logger.debug(
            '{} is flattening inventory of {}'.format(self.side, self.position_count))

        if self.position_count < 1:
            return -0.00000000001

        pnl = 0.
        while self.position_count > 0:
            pnl += self.remove(price=price)

        return pnl

    def get_distance_to_midpoint(self, midpoint: float):
        """
        Distance percentage between current midpoint price and the open order's
        posted price.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) distance between open order and midpoint price
        """
        if self.order is None:
            return 0.
        return abs(midpoint - self.order.price) / self.order.price
