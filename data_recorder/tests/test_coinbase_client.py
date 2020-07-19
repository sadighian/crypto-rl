import asyncio

from data_recorder.coinbase_connector.coinbase_client import CoinbaseClient

if __name__ == "__main__":
    """
    This __main__ function is used for testing the
    CoinbaseClient class in isolation.
    """

    loop = asyncio.get_event_loop()
    symbols = ['BTC-USD']  # 'LTC-USD', 'ETH-USD']
    p = dict()

    print('Initializing...%s' % symbols)
    for sym in symbols:
        p[sym] = CoinbaseClient(sym=sym)
        p[sym].start()

    tasks = asyncio.gather(*[(p[sym].subscribe()) for sym in symbols])
    print('Gathered %i tasks' % len(symbols))

    try:
        loop.run_until_complete(tasks)
        print('TASK are complete for {}'.format(symbols))
        loop.close()
        for sym in symbols:
            p[sym].join()
            print('Closing [%s]' % p[sym].name)
        print('loop closed.')

    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        loop.close()
        for sym in symbols:
            p[sym].join()
            print('Closing [%s]' % p[sym].name)

    finally:
        loop.close()
        print('\nFinally done.')
