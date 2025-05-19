# cyclic-process-forecasting
Pattern-matching forecasts for cyclic time series using DTW and Euclidean distance methods


Cyclic Time Series Forecasting Tool
------------------------------------

This Python script implements a non-parametric forecasting method for cyclic time series using pattern matching.
It forecasts the remainder of a cycle by identifying the most similar segments from previous cycles using either:

- Euclidean distance
- Dynamic Time Warping (DTW)

The approach does not rely on training a model. Instead, it uses historical cycle shapes as a dictionary and forecasts
based on their continuation behavior.

Core Features:
--------------
- Automatically segments the signal into individual cycles using valley detection
- Matches the current partial cycle against a library of previous cycles
- Supports bounded sliding window matching within a ±range
- Forecasts the future trajectory using the top-1, top-2, and top-3 neighbors
- Computes:
    - Forecast accuracy (RMSE)
    - End-of-cycle time prediction error (e.g., when signal crosses a value)
- Provides both single-shot visualization and multi-start evaluation plots

Key Functions:
--------------

1. get_clipped_cycles(...):
   Segments a univariate time series into individual cycles using valleys (local minima).

   Parameters:
   - min_distance: Minimum distance between valleys (default = 100)
   - min_prominence: Required drop from neighboring peaks to count as a valley (default = 2)

2. forecast_with_bounded_sliding_method(...):
   Finds similar windows in previous cycles based on a sliding search.
   Forecasts are generated using:
   - Top-1 nearest match
   - Weighted average of top-2 and top-3 matches

   Supports:
   - method='euclidean'
   - method='dtw'
   -  search_start = max(0, forecast_start - window_size - 20)
      search_end = min(len(hist) - window_size, forecast_start - window_size + 20)
      +-20 here specifies segments to search over cycles in the data set using the current time point in the running cycle as the reference

3. run_single_forecast_plot(...):
   Performs a forecast on a single test cycle at a specified time step, and plots:
   - The full test cycle
   - The forecast window (shaded)
   - Forecasts from 3 neighbors × each method

4. run_forecast_multiple_starts(...):
   Runs forecasts at multiple time points to evaluate performance as more data becomes available.

   Plots:
   - Average RMSE vs forecast start
   - End time prediction error vs forecast start

User-Specified Inputs:
----------------------

| Parameter            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| file_path            | Excel file path with the time series                                        |
| sheet_name           | Excel sheet name                                                            |
| target_col_index     | Index of the column to analyze (e.g., 0 for synthetic example)              |
| test_cycle_index     | Which cycle to forecast (0-indexed)                                         |
| forecast_start       | Time step within test cycle to start forecasting                            |
| window_size          | Number of steps to use as the lookback window                               |
| threshold            | Temperature value to define end-of-cycle (e.g., 40°C for crossing detection)|
| methods              | List of distance methods: ['euclidean'], ['dtw'], or both                   |

Tuning Parameters:
------------------

| Parameter           | Role                                                                 |
|---------------------|----------------------------------------------------------------------|
| window_size         | Length of the input segment used for pattern matching                |
| forecast_start      | Point within the test cycle to begin forecasting                     |
| min_distance        | Minimum distance between valleys for cycle segmentation              |
| min_prominence      | Controls how deep valleys must be to count as cycle boundaries       |
| sliding window ±    | Search range within historical cycles (currently hardcoded ±20 steps)|
| top-K neighbors     | Fixed at 3 for now (can be generalized)                              |

Example Usage:
--------------

To forecast cycle 29 starting from time step 50 using a window of 20:

run_single_forecast_plot(
    file_path="Synthetic_Cyclic_Data.xlsx",
    sheet_name="SyntheticCyclicExample",
    test_cycle_index=29,
    forecast_start=50,
    window_size=20,
    target_col_index=0,
    threshold=40,
    methods=['euclidean', 'dtw']
)

