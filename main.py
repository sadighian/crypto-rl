import asyncio
import os
from multiprocessing import Process
from bitfinex_connector.BitfinexClient import BitfinexClient
from gdax_connector.GdaxClient import GdaxClient


symbols = [['BTC-USD', 'tBTCUSD'],#,  # bitcoin
           ['ETH-USD', 'tETHUSD'],  # ether
           ['BCH-USD', 'tBCHUSD'],  # bitcoin cash
           ['LTC-USD', 'tLTCUSD']]  # litecoin


def create_workers(basket, index):
    """
    Create a map of threads to currencies, 1-to-1 relationship
    :param basket: currencies being traded (e.g., bitcoin, ether, ...)
    :param index: gdax = 0, bitfinex = 1
    :return: dictionary of connector clients (gdax or bitfinex)
    """
    workers = dict()
    for sym in basket:
        workers[sym[index]] = GdaxClient(sym[index]) if index == 0 else BitfinexClient(sym[index])
        workers[sym[index]].start()
        print('[%s] started for [%s] with process_id %s' % (workers[sym[index]].name, sym[index], str(os.getpid())))
    return workers


def gdax_process():
    """
    Process spawned to connected to gdax exchange
    :return: void
    """
    print('invoking gdax_process %s\n' % str(os.getpid()))

    workers = create_workers(symbols, index=0)
    tasks = asyncio.gather(*[workers[sym].subscribe() for sym in workers.keys()])
    loop = asyncio.get_event_loop()
    print('gdax_process Gathered %i tasks' % len(workers.keys()))

    try:
        loop.run_until_complete(tasks)
        loop.close()
        print('gdax_process loop closed.')

    except KeyboardInterrupt as e:
        print("gdax_process Caught keyboard interrupt. Canceling tasks... %s" % e)
        tasks.cancel()
        tasks.exception()
        for sym in symbols:
            workers[sym].unsubscribe()  # unsubscribe from websocket (does not apply to bitfinex)
            workers[sym].join()
            print('gdax_process Closing [%s]' % workers[sym].name)

    finally:
        loop.close()
        print('\ngdax_process Finally done.')


def bitfinex_process():
    """
    Process spawned to connect to bitfinex exchange
    :return: void
    """
    print('invoking bitfinex_process %s\n' % str(os.getpid()))

    p = create_workers(symbols, index=1)
    tasks = asyncio.gather(*[p[sym].subscribe() for sym in p.keys()])
    loop = asyncio.get_event_loop()
    print('bitfinex_process Gathered %i tasks' % len(p.keys()))

    try:
        loop.run_until_complete(tasks)
        loop.close()
        print('bitfinex_process loop closed.')

    except KeyboardInterrupt as e:
        print("bitfinex_process Caught keyboard interrupt. Canceling tasks... %s" % e)
        tasks.cancel()
        tasks.exception()
        for sym in symbols:
            p[sym].join()
            print('bitfinex_process Closing [%s]' % p[sym].name)

    finally:
        loop.close()
        print('\nbitfinex_process Finally done.')


# ----------------------------------------------------------------------------------------

# def do_main():
#     print('invoking do_main() on %s\n' % str(os.getpid()))
#
#     basket = [['BTC-USD'],  # ,  # bitcoin
#                ['tBTCUSD']]  # ether
#
#     workers = dict()
#     for gdax, bitfinex in zip(*basket):
#         workers[gdax], workers[bitfinex] = GdaxClient(gdax), BitfinexClient(bitfinex)
#         workers[gdax].start(), workers[bitfinex].start()
#         print('[%s] started for [%s] with process_id %s' % (gdax, bitfinex, str(os.getpid())))
#
#     tasks = asyncio.gather(*[workers[sym].subscribe() for sym in workers.keys()])
#     loop = asyncio.get_event_loop()
#     print('gdax_process Gathered %i tasks' % len(workers.keys()))
#
#     try:
#         loop.run_until_complete(tasks)
#         loop.close()
#         print('gdax_process loop closed.')
#
#     except KeyboardInterrupt as e:
#         print("gdax_process Caught keyboard interrupt. Canceling tasks... %s" % e)
#         tasks.cancel()
#         tasks.exception()
#
#     finally:
#         loop.close()
#         print('\ngdax_process Finally done.')


if __name__ == "__main__":
    print('Starting up...__main__ Process ID: %s\n' % str(os.getpid()))
    # do_main()
    p_gdax, p_bitfinex = Process(target=gdax_process), Process(target=bitfinex_process)
    p_gdax.start(), p_bitfinex.start()

