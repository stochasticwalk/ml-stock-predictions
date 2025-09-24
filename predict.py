import json
import os
import pandas as pd
from joblib import load


def predict_next_week(start_csv: str, end_csv: str, tickers_json: str = 'used_tickers.json',
                      models_dir: str = 'models', output_csv: str = 'predictions.csv') -> pd.DataFrame:
    """
    Loads two .csv files (one from the first day of a trading week, one from the last day)
    Predicts the next week's category label (1-7, higher is better)

    :param start_csv: First trading day .csv
    :param end_csv:  Last trading day .csv
    :param tickers_json: Saved list of tickers used for model training
    :param models_dir: Location of the saved models
    :param output_csv: Filename for the output
    :return: Returns a DataFrame to inspect (but the .csv file is for actual decision-making)
    """

    # Load CSV files
    start_df = pd.read_csv(start_csv)
    end_df = pd.read_csv(end_csv)

    if start_df.empty or end_df.empty:
        print("Empty start or end CSV. Returning empty results.")
        return pd.DataFrame()

    # Load used tickers and filter DataFrames
    with open(tickers_json, 'r') as f:
        used_tickers = json.load(f)
    used_tickers = sorted(used_tickers)  # Ensure sorted ascending

    # Filter to used_tickers, reindex to include all (add NaN rows for missing)
    ticker_to_row_start = {row['ticker']: row for _, row in start_df.iterrows()}
    ticker_to_row_end = {row['ticker']: row for _, row in end_df.iterrows()}

    start_rows = []
    end_rows = []
    for ticker in used_tickers:
        if ticker in ticker_to_row_start:
            start_rows.append(ticker_to_row_start[ticker])
        else:
            # Add NaN row (copy structure, but NaNs)
            nan_row_start = pd.Series(index=start_df.columns, dtype=object)
            nan_row_start['ticker'] = ticker
            nan_row_start['date'] = start_df['date'].iloc[0] if not start_df.empty else pd.NaT
            start_rows.append(nan_row_start)

        if ticker in ticker_to_row_end:
            end_rows.append(ticker_to_row_end[ticker])
        else:
            nan_row_end = pd.Series(index=end_df.columns, dtype=object)
            nan_row_end['ticker'] = ticker
            nan_row_end['date'] = end_df['date'].iloc[0] if not end_df.empty else pd.NaT
            end_rows.append(nan_row_end)

    start_df = pd.DataFrame(start_rows).reset_index(drop=True)
    end_df = pd.DataFrame(end_rows).reset_index(drop=True)
    start_df = start_df.sort_values('ticker').reset_index(drop=True)
    end_df = end_df.sort_values('ticker').reset_index(drop=True)

    # Step 3: Create change DataFrame
    change_df = pd.DataFrame({'ticker': used_tickers})

    # Compute change: (end.adj_close - start.adj_open) / end.adj_close
    mask_no_nan = ~(start_df['adj_open'].isna() | end_df['adj_close'].isna()) & (end_df['adj_close'] != 0)
    change_df['change'] = 0.0  # Default
    valid_changes = (end_df.loc[mask_no_nan, 'adj_close'] - start_df.loc[mask_no_nan, 'adj_open']) / end_df.loc[
        mask_no_nan, 'adj_close']
    change_df.loc[mask_no_nan, 'change'] = valid_changes.values

    # Prepare feature array (same format as training: 1D array of changes)
    change_array = change_df['change'].values

    # Initialize results_dict
    results_dict = {}

    # Load models and predict
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
    print(f"Found {len(model_files)} model files in {models_dir}.")

    for model_file in model_files:
        try:
            ticker = model_file.replace('_model.joblib', '')  # Extract ticker name
            if ticker not in used_tickers:
                continue  # Skip if not in used tickers

            model_path = os.path.join(models_dir, model_file)
            model = load(model_path)

            # Predict (returns array, take [0] for single prediction)
            label = int(model.predict(change_array)[0])
            results_dict[ticker] = label

        except Exception as e:
            print(f"Error loading/predicting with {model_file}: {e}")
            continue

    # Convert to DataFrame and save as .csv
    if results_dict:
        results_df = pd.DataFrame(list(results_dict.items()), columns=['ticker', 'predicted_label'])
        results_df = results_df.sort_values('ticker').reset_index(drop=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved {len(results_df)} predictions to {output_csv}.")
    else:
        results_df = pd.DataFrame(columns=['ticker', 'predicted_label'])
        print("No predictions made (no valid models). Empty CSV not saved.")

    return results_df

if __name__ == "__main__":
    results = predict_next_week('week_start.csv', 'week_end.csv')
    print(results.head(10))