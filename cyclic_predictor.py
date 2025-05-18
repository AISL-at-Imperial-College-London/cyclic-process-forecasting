# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from dtaidistance import dtw
from scipy.signal import find_peaks

# --- Helper Functions ---
def rmse(pred, true):
    return np.sqrt(mean_squared_error(true, pred))

def get_clipped_cycles(signal, min_distance=100, min_prominence=2):
    inverted_signal = -signal
    valley_indices, _ = find_peaks(inverted_signal, distance=min_distance, prominence=min_prominence)

    print(f"Detected {len(valley_indices)-1} cycles.")

    if len(valley_indices) < 2:
        raise ValueError("Not enough valleys detected to segment cycles.")

    cycle_segments = [signal[valley_indices[i]:valley_indices[i+1]] for i in range(len(valley_indices)-1)]
    min_len = min(len(c) for c in cycle_segments)
    return np.array([c[:min_len] for c in cycle_segments])

def weighted_average(forecasts, distances):
    weights = 1 / (distances + 1e-8)
    weights /= weights.sum()
    return np.average(forecasts, axis=0, weights=weights)

def forecast_with_bounded_sliding_method(test_cycle, historical_cycles, window_size, forecast_start, method='euclidean'):
    test_window = test_cycle[forecast_start - window_size:forecast_start]
    distances = []
    starts = []
    for hist in historical_cycles:
        best_dist = float('inf')
        best_start = 0
        search_start = max(0, forecast_start - window_size - 20)
        search_end = min(len(hist) - window_size, forecast_start - window_size + 20)
        for i in range(search_start, search_end + 1):
            hist_window = hist[i:i+window_size]
            if method == 'euclidean':
                dist = np.linalg.norm(test_window - hist_window)
            elif method == 'dtw':
                dist = dtw.distance(test_window, hist_window)
            else:
                raise ValueError("Unknown method. Use 'euclidean' or 'dtw'.")
            if dist < best_dist:
                best_dist = dist
                best_start = i + window_size
        distances.append(best_dist)
        starts.append(best_start)

    distances = np.array(distances)
    nearest_indices = np.argsort(distances)[:3]
    forecasts = []
    for idx in nearest_indices:
        start = starts[idx]
        forecast = historical_cycles[idx][start:]
        if len(forecast) >= len(test_cycle) - forecast_start:
            forecasts.append(forecast[:len(test_cycle) - forecast_start])
        else:
            forecasts.append(np.pad(forecast, (0, len(test_cycle) - forecast_start - len(forecast)), 'edge'))

    forecasts = np.array(forecasts)
    pred_1 = forecasts[0]
    pred_2 = weighted_average(forecasts[:2], distances[nearest_indices[:2]])
    pred_3 = weighted_average(forecasts, distances[nearest_indices])
    cycle_ids = [i for i in nearest_indices]
    return pred_1, pred_2, pred_3, distances[nearest_indices], cycle_ids

def find_crossing_index(series, threshold=40):
    for i in range(1, len(series)):
        if series[i-1] > threshold and series[i] <= threshold:
            return i
    return None

# Single forecast and plot the time series forecast with shading
def run_single_forecast_plot(
    file_path,
    sheet_name,
    test_cycle_index=9,
    forecast_start=300,
    window_size=100,
    target_col_index=29,
    threshold=40,
    methods=['euclidean', 'dtw']
):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
    signal = pd.to_numeric(df.iloc[:, target_col_index], errors='coerce').dropna().reset_index(drop=True)
    all_cycles = get_clipped_cycles(signal.values)
    test_cycle = all_cycles[test_cycle_index]
    historical_cycles = [c for i, c in enumerate(all_cycles) if i != test_cycle_index]

    if forecast_start >= len(test_cycle) or forecast_start - window_size < 0:
        raise ValueError("Forecast start out of bounds.")

    true_future = test_cycle[forecast_start:]
    colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#0072B2']
    linestyles = ['--', '--', '--', '-.', '-.', '-.']
    labels = []

    plt.figure(figsize=(14, 6))
    plt.plot(test_cycle, color='black', label='Actual Test Cycle', linewidth=2)
    plt.axvspan(forecast_start - window_size, forecast_start, color='gray', alpha=0.15, label='Lookback Window')

    method_count = 0
    for method in methods:
        try:
            p1, p2, p3, dists, ids = forecast_with_bounded_sliding_method(
                test_cycle, historical_cycles, window_size, forecast_start, method=method
            )
            for i, pred in enumerate([p1, p2, p3]):
                plt.plot(
                    np.arange(forecast_start, len(test_cycle)),
                    pred,
                    linestyle=linestyles[method_count + i],
                    color=colors[method_count + i],
                    label=f'{method.upper()} ({i+1})'
                )
        except Exception as e:
            print(f"Warning: {method} forecast failed — {e}")
        method_count += 3

    plt.axvline(x=forecast_start, color='gray', linestyle=':', label=f'Prediction Start @ Step {forecast_start}')
    plt.title(f"Forecast from Step {forecast_start} with Window Size {window_size}")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Main Forecast Evaluation Function ---
def run_forecast_multiple_starts(
    file_path,
    sheet_name,
    test_cycle_index=9,
    window_size=100,
    forecast_starts=[50, 60, 70, 80, 90, 100, 110],
    target_col_index=29,
    threshold=40,
    methods=['euclidean', 'dtw']
):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
    signal = pd.to_numeric(df.iloc[:, target_col_index], errors='coerce').dropna().reset_index(drop=True)
    all_cycles = get_clipped_cycles(signal.values)
    test_cycle = all_cycles[test_cycle_index]
    historical_cycles = [c for i, c in enumerate(all_cycles) if i != test_cycle_index]

    results = []

    for forecast_start in forecast_starts:
        if forecast_start >= len(test_cycle) or forecast_start - window_size < 0:
            print(f"Skipping forecast_start={forecast_start} due to bounds.")
            continue

        true_future = test_cycle[forecast_start:]
        for method in methods:
            print(f"Running {method.upper()} @ step {forecast_start}...")
            try:
                p1, p2, p3, dists, ids = forecast_with_bounded_sliding_method(
                    test_cycle, historical_cycles, window_size, forecast_start, method=method
                )
                for i, pred in enumerate([p1, p2, p3]):
                    rmse_val = rmse(pred, true_future)
                    cross_pred = find_crossing_index(pred, threshold)
                    cross_true = find_crossing_index(true_future, threshold)
                    cross_error = (cross_pred - cross_true) if (cross_pred is not None and cross_true is not None) else None
                    results.append({
                        'ForecastStart': forecast_start,
                        'Method': f'{method.upper()} ({i+1})',
                        'Avg RMSE': rmse_val,
                        'End Time Error': cross_error
                    })
            except Exception as e:
                print(f"⚠️  Warning: Method {method} failed at start {forecast_start} — {e}")
                continue

    df_results = pd.DataFrame(results)

    # --- Plot 1: End Time Error ---
    plt.figure(figsize=(10, 5))
    for method in df_results['Method'].unique():
        subset = df_results[df_results['Method'] == method]
        plt.plot(subset['ForecastStart'], subset['End Time Error'], label=method, marker='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('End Time Prediction Error vs. Forecast Start')
    plt.xlabel('Forecast Start Time')
    plt.ylabel('End Time Error (steps)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: RMSE ---
    plt.figure(figsize=(10, 5))
    for method in df_results['Method'].unique():
        subset = df_results[df_results['Method'] == method]
        plt.plot(subset['ForecastStart'], subset['Avg RMSE'], label=method, marker='o')
    plt.title('Average RMSE vs. Forecast Start')
    plt.xlabel('Forecast Start Time')
    plt.ylabel('Avg RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_results

# results_df = run_forecast_multiple_starts(
#     file_path="Synthetic_Cyclic_Data.xlsx",
#     sheet_name="SyntheticCyclicExample",
#     test_cycle_index=9,
#     forecast_start=50,     # <- change this to any step you want
#     window_size=50,
#     target_col_index=0,
#     threshold=40,
#     methods=['euclidean', 'dtw']  # Enables both algorithms
# )

run_single_forecast_plot(
    file_path="Synthetic_Cyclic_Data.xlsx",
    sheet_name="SyntheticCyclicExample",
    test_cycle_index=29,
    forecast_start=50,     # <- change this to any step you want
    window_size=20,
    target_col_index=0,
    threshold=40,
    methods=['euclidean', 'dtw']  # both methods will be shown in the plot
)

