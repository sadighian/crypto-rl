# Database
As of September 13, 2019.

## 1. Overview
The `database` module contains the wrapper class for storing tick data from the `Arctic Tick Store`.

**Note** the `../simulator.py` class is used to query and reconstruct the limit order book.

## 2. Classes

### 2.1 Database
This is a wrapper class used for storing streaming tick data into the `Arctic Tick Store`. 
The `new_tick()` method is implemented in both `bifinex_connector` and 
`coinbase_connector` projects.

### 2.2 Simulator
This is a utility class to query Arctic (database), replay historical
data, and export order book snapshots to a compressed csv.

### 2.3 Viz
This is a utility class to plot the features data exported from
`simulator.py`

Example diagrams are below:

![plot_lob_overlay](../../design_patterns/plot_lob_overlay.PNG)

![plot_lob_levels](../../design_patterns/plot_lob_levels.PNG)

![plot_transactions](../../design_patterns/plot_transactions.PNG)

![plot_order_arrivals](../../design_patterns/plot_order_arrivals.PNG)
 

