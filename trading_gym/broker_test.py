from broker import Broker


def test_case_one():
    print('\nTest_Case_One')
    params = {
        'ccy': 'BTC-USD',
        'max_position': 1
    }
    test_position = Broker(**params)
    midpoint = 100.
    fee = .003

    # ========

    order_open = {
        'price': midpoint + (midpoint * fee),
        'size': 10000.,
        'side': 'long'
    }
    test_position.add(order=order_open)

    assert test_position.long_inventory.position_count == 1
    print('LONG Unrealized_pnl: %f' % test_position.long_inventory.get_unrealized_long_pnl())

    assert test_position.short_inventory.position_count == 0
    assert test_position.short_inventory.get_unrealized_long_pnl() == 0.

    order_close = {
        'price': (midpoint * fee) * 5. + midpoint,
        'size': 10000.,
        'side': 'long'
    }
    test_position.remove(order=order_close)
    assert test_position.long_inventory.position_count == 0
    print('LONG Unrealized_pnl: %f' % test_position.long_inventory.get_unrealized_long_pnl())

    assert test_position.short_inventory.position_count == 0
    assert test_position.short_inventory.get_unrealized_long_pnl() == 0.
    print('LONG Realized_pnl: %f' % test_position.get_realized_pnl())


def test_case_two():
    print('\nTest_Case_Two')
    params = {
        'ccy': 'LTC-USD',
        'max_position': 1
    }
    test_position = Broker(**params)
    midpoint = 100.
    fee = .003

    # ========

    order_open = {
        'price': midpoint - (midpoint * fee),
        'size': 10000.,
        'side': 'short'
    }
    test_position.add(order=order_open)

    assert test_position.short_inventory.position_count == 1
    print('SHORT Unrealized_pnl: %f' % test_position.short_inventory.get_unrealized_short_pnl())

    assert test_position.long_inventory.position_count == 0
    assert test_position.long_inventory.get_unrealized_long_pnl() == 0.

    order_close = {
        'price': midpoint * 0.95,
        'size': 10000.,
        'side': 'short'
    }
    test_position.remove(order=order_close)
    assert test_position.short_inventory.position_count == 0
    print('SHORT Unrealized_pnl: %f' % test_position.short_inventory.get_unrealized_short_pnl())

    assert test_position.long_inventory.position_count == 0
    assert test_position.long_inventory.get_unrealized_long_pnl() == 0.
    print('SHORT Realized_pnl: %f' % test_position.get_realized_pnl())


def test_case_three():
    print('\nTest_Case_Three')
    params = {
        'ccy': 'LTC-USD',
        'max_position': 5
    }
    test_position = Broker(**params)
    midpoint = 100.
    fee = .003

    # ========

    for i in range(10):
        order_open = {
            'price': midpoint + i,
            'size': 10000.,
            'side': 'long'
        }
        test_position.add(order=order_open)

    print('Confirm we have 5 positions: %i' % test_position.long_inventory.position_count)
    assert test_position.long_inventory.position_count == 5
    assert test_position.short_inventory.position_count == 0

    for i in range(10):
        order_open = {
            'price': midpoint - i,
            'size': 10000.,
            'side': 'long'
        }
        test_position.remove(order=order_open)

    assert test_position.long_inventory.position_count == 0
    assert test_position.short_inventory.position_count == 0

if __name__ == "__main__":
    """
    Entry point of application
    """
    test_case_one()
    test_case_two()
    test_case_three()
