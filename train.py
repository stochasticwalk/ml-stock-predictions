import pandas as pd
import json
import os
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

def prepare_ml_data(csv_path: str, market_threshold: float) -> pd.DataFrame:
    """
    Outputs a DataFrame with the necessary data for machine learning

    :param csv_path: The .csv file that resulted from preproc.py
    :param market_threshold: The final week's end date 'adj_close' * 'adj_volume', filters out penny stocks
    :return: ML-ready DataFrame with columns 'ticker', 'week_num', 'change', and 'category_label', one row per ticker-week, sorted ascending
    """
    # Load .csv
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Input .csv is empty, returning empty DataFrame")
        return pd.DataFrame()

    max_week = df['week_num'].max()

    #Filter based on market threshold
    final_week_df = df[(df['week_num'] == max_week) & (df['start_end'] == 'end')]
    if final_week_df.empty:
        print("No 'end' rows in final week, returning empty DataFrame")
        return pd.DataFrame()

    final_week_df['market'] = final_week_df['adj_close'] * final_week_df['adj_volume']
    valid_tickers = final_week_df[final_week_df['market'] >= market_threshold]['ticker'].unique()
    if len(valid_tickers) == 0:
        print("No qualifying tickers, returning empty DataFrame")
        return pd.DataFrame()

    df = df[df['ticker'].isin(valid_tickers)].copy()

    # Separate start and end data, merge so one row per ticker-week, compute change and categorize
    start_df = df[df['start_end'] == 'start'][['ticker', 'week_num', 'adj_open']].copy()
    end_df = df[df['start_end'] == 'end'][['ticker', 'week_num', 'adj_close']].copy()
    if start_df.empty or end_df.empty:
        print("No start or end rows found, returning empty DataFrame")
        return pd.DataFrame()

    ml_df = start_df.merge(end_df, on=['ticker', 'week_num'], how='inner')
    ml_df = ml_df.dropna().copy()
    if ml_df.empty:
        print("No complete week pairings, returning empty DataFrame")
        return pd.DataFrame()

    ml_df['change'] = (ml_df['adj_close'] - ml_df['adj_open']) / ml_df['adj_close']
    ml_df['change'] = ml_df['change'].fillna(0.0)
    ml_df = ml_df[['ticker', 'week_num', 'change']].copy()

    # Bin the changes into 7 equally sized categories and label accordingly
    ml_df['category_label'] = pd.qcut(ml_df['change'], q=7, labels=[1, 2, 3, 4, 5, 6, 7])
    ml_df['category_label'] = ml_df['category_label'].astype(int)
    ml_df = ml_df.sort_values(['ticker', 'week_num']).reset_index(drop=True)

    print("ML DataFrame created")
    print(f"Label distribution: {ml_df['category_label'].value_counts().sort_index().to_dict()}")
    return ml_df

def train_models(df: pd.DataFrame, accuracy_threshold: float=0.5, model_dir: str='models',
                 tickers_json: str='used_tickers.json') -> None:
    """
    Trains Random Forest Classifiers per ticker in the DataFrame from prepare_ml_data
    Uses week 'n' change values to try and predict week 'n+1' category for that ticker
    If the accuracy of the model exceeds the threshold, it is saved to a folder for later use
    Custom accuracy scoring: 1.0 - 0.5 * |pred - true|, clamped to [-0.5, 1.0] per prediction, then averaged across testing

    :param df: The DataFrame from prepare_ml_data with 'ticker', 'week_num', 'change', and 'category_label'
    :param accuracy_threshold: Minimum accuracy threshold for model to be saved
    :param model_dir: Location to save the valid models
    :param tickers_json: Filename to save the tickers used, important so future data has the same shape and order as training data
    """
    if df.empty:
        print("No input DataFrame, no training performed")
        return

    # Prepare the ticker list and model file directory
    unique_tickers = sorted(df['ticker'].unique())
    if not unique_tickers:
        print("No tickers found, no training performed")
        return

    with open(tickers_json, 'w') as f:
        json.dump(unique_tickers, f)

    os.makedirs(model_dir, exist_ok=True)

    # Pivot the DataFrame data for features
    # Make sure values are sorted as expected
    df = df.sort_values(['week_num', 'ticker']).copy()
    changes_pivot = df.pivot(index='week_num', columns='ticker', values='change').loc[:, unique_tickers].fillna(0)
    weeks = sorted(changes_pivot.index.tolist())
    num_weeks = len(weeks)
    if num_weeks < 2:
        print("Not enough weeks for training, no training performed")
        return

    # Make the prediction indices from week 1 to num_weeks-1 since we're trying to predict a week ahead
    pred_indices = range(1, num_weeks)
    # Split 80/20 training/testing
    split_idx = int(len(pred_indices) * 0.8)
    train_start = 1
    train_end = split_idx+1
    test_start = split_idx+1
    test_end = num_weeks

    # X_train from weeks 0 to split_idx-1
    X_train = changes_pivot.iloc[:split_idx].values
    # X_test from weeks split_idx to num_weeks-2
    X_test = changes_pivot.iloc[split_idx: num_weeks - 1].values
    test_num_samples = X_test.shape[0]
    if split_idx == 0 or test_num_samples == 0:
        print("Insufficient weeks, no training performed")
        return

    # Custom private accuracy function, uses numpy array outputs from training/testing
    def custom_accuracy(y_true, y_guess):
        errors = np.abs(y_true - y_guess)
        scores = np.clip(1.0 - 0.5 * errors, -0.5, 1.0)
        return np.mean(scores)

    # Iterate over tickers and train, save accuracy scores to a dict/json
    accuracy_scores = {}
    saved_count = 0
    for i, ticker in enumerate(unique_tickers):
        print(f"Training model {i + 1}/{len(unique_tickers)}: {ticker}")

        # Extract y for this ticker (category_label, indexed by week_num)
        ticker_df = df[df['ticker'] == ticker].set_index('week_num')['category_label'].sort_index()
        y_train = ticker_df.iloc[1:train_end].values
        y_test = ticker_df.iloc[test_start:test_end].values

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute accuracy of the current model
        score = custom_accuracy(y_test, y_pred)
        print(f"  Test accuracy (custom): {score:.3f}")

        # Save if above the specified accuracy threshold
        if score >= accuracy_threshold:
            model_path = os.path.join(model_dir, f"{ticker}_model.joblib")
            dump(model, model_path)
            print(f"  Saved model to {model_path}")
            accuracy_scores[ticker] = score
            saved_count += 1
        else:
            print(f"  Below threshold ({score:.3f} < {accuracy_threshold}). Discarded.")

    # Export the accuracy scores of the model to a separate .csv for future examination
    acc_df = pd.DataFrame(list(accuracy_scores.items()), columns=['ticker', 'accuracy'])
    acc_df.to_csv('model_accuracy.csv', index=False)

    print(f"Training complete. Saved {saved_count}/{len(unique_tickers)} models.")

if __name__ == "__main__":
    mkt_threshold = 10000000.0 # Threshold to filter out penny stocks
    training_df = prepare_ml_data('weekly_prices.csv', mkt_threshold)
    train_models(training_df)