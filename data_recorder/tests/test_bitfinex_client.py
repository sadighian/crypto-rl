import asyncio

from data_recorder.bitfinex_connector.bitfinex_client import BitfinexClient

if __name__ == "__main__":
    """
    This __main__ function is used for testing the
    BitfinexClient class in isolation.
    """
    symbols = ['tBTCUSD']  # 'tETHUSD', 'tLTCUSD']
    print('Initializing...%s' % symbols)
    loop = asyncio.get_event_loop()
    p = dict()

    for sym in symbols:
        p[sym] = BitfinexClient(sym=sym)
        p[sym].start()
        print('Started thread for %s' % sym)

    tasks = asyncio.gather(*[(p[sym].subscribe()) for sym in symbols])
    print('Gathered %i tasks' % len(symbols))

    try:
        loop.run_until_complete(tasks)
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
