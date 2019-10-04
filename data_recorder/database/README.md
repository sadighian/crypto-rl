# Database
As of September 26, 2019.

## 1. Overview
The `database` module contains the wrapper class for storing tick data
from the `Arctic Tick Store`.

**Note** the `../simulator.py` class is used to reconstruct the limit
order book.

## 2. Classes

### 2.1 Database
This is a wrapper class used for storing streaming tick data into the
`Arctic Tick Store`.

-  The `new_tick()` method is used to persist data to Arctic and it is
   implemented in both `bifinex_connector` and `coinbase_connector`
   projects.
-  The `get_tick_history` method is used to query Arctic and return its
   `cursor` in the form of a `pd.DataFrame`; it is implemented in
   `database.py`

### 2.2 Simulator
This is a utility class to replay historical data, and export order book
snapshots to a xz-compressed csv.

An example of how to export LOB snapshots to a csv is below:

```
# Find LTC-USD ticks from Coinbase Pro between April 06, 2019 and April 07, 2019.
query = {
        'ccy': ['LTC-USD'],
        'start_date': 20190406,
        'end_date': 20190407
    }
    
    
# Or, find LTC-USD ticks from Coinbase Pro AND Bitfinex exchange between April 06, 2019 and April 07, 2019.
query = {
        'ccy': ['LTC-USD', 'tLTCUSD'],
        'start_date': 20190406,
        'end_date': 20190407
    }
    
    
sim = Simulator()
sim.extract_features(query)

# Done !
```

### 2.3 Viz
This is a utility class to plot the features data exported from
`simulator.py`

Example diagrams using 500-millisecond snapshots of ETH-USD's limit
order book are below:

`plot_lob_overlay`
![plot_lob_overlay](../../design_patterns/plot_lob_overlay.png)

`plot_lob_levels`
![plot_lob_levels](../../design_patterns/plot_lob_levels.png)

`plot_transactions`
![plot_transactions](../../design_patterns/plot_transactions.png)

`plot_order_arrivals`
![plot_order_arrivals](../../design_patterns/plot_order_arrivals.png)
 

