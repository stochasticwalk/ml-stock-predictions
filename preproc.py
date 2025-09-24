import pandas as pd

def reduce_csv(file_path: str, start_date: str = '2012-01-01') -> pd.DataFrame:
    """
    Imports a .csv file from the NASDAQ EOD data product and reduces to a useful DataFrame
    Filters to dates >= start_date, selects only tickers that have complete data within the date range
    Optimized for low-spec hardware (large .csv file has large RAM reqs)

    :param file_path: The file path/name of the .csv
    :param start_date: The earliest date you want in the dataset
    :return: A smaller dataframe with select tickers and a desired date range
    """
    # Load the .csv with date parsing, selected columns, and optimized dtypes
    dtype_dict = {
        'ticker': 'category',
        'adj_open': 'float32',
        'adj_close': 'float32',
        'adj_volume': 'int64'
    }
    start_dt = pd.to_datetime(start_date)
    chunk_size = 50000
    chunks = []

    use_cols = ['ticker', 'date', 'adj_open', 'adj_close', 'adj_volume']
    print("Loading and filtering file in chunks...")
    for chunk in pd.read_csv(
        file_path,
        parse_dates=['date'],
        dtype=dtype_dict,
        chunksize=chunk_size,
        usecols=use_cols
    ):
        # Filter dates in this chunk
        filtered_chunk = chunk[chunk['date'] >= start_dt]
        if not filtered_chunk.empty:
            chunks.append(filtered_chunk)

    if not chunks:
        print("No data after start date, returning empty DataFrame")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Get the unique dates in the filtered data
    unique_dates = df['date'].unique()
    num_unique_dates = len(unique_dates)

    # Identify tickers with a complete dataset in the date range
    ticker_counts = df.groupby('ticker').size()
    valid_tickers = ticker_counts[ticker_counts == num_unique_dates].index.tolist()

    if not valid_tickers:
        print("No tickers found with complete data, returning empty DataFrame")
        return pd.DataFrame()

    # Filter the DataFrame so it contains only the valid tickers
    df = df[df['ticker'].isin(valid_tickers)]

    # Sort by date, then by ticker (ascending)
    df = df.sort_values(['date', 'ticker'], ascending=[True, True]).reset_index(drop=True)

    print("Preprocessing complete!")
    return df

def reduce_to_weeks(df: pd.DataFrame, output_path: str = 'weekly_prices.csv') -> pd.DataFrame:
    """
    Adds week data to the processed price DataFrame, for use in training
    Adds a week number and a start/end flag, then filters the DataFrame to only contain these rows
    Returns a DataFrame for inspection, but also writes to a .csv file

    :param df: The resulting DataFrame from reduce_csv
    :param output_path: Path to save the final .csv (default 'weekly_prices.csv')
    :return: Filtered DataFrame with week numbers and only rows flagged as a week start/end
    """
    if df.empty:
        print("Input DataFrame is empty, returning empty DataFrame")
        return df

    # Create a separate DataFrame to do date -> weekday calculations
    date_df = pd.DataFrame({'date': df['date'].unique()}).sort_values('date').reset_index(drop=True)

    # Add a weekday column (0 = Monday, 4 = Friday, etc.)
    date_df['weekday'] = date_df['date'].dt.weekday

    # Initialize columns for start/end flags and week numbering
    date_df['start_end'] = pd.NA
    date_df['week_num'] = -1

    # Add shift calculations for week boundary detection
    date_df['prev_weekday'] = date_df['weekday'].shift(1)
    date_df['next_weekday'] = date_df['weekday'].shift(-1)

    # Flag start and end by column comparison
    date_df.loc[date_df['weekday'] < date_df['prev_weekday'], 'start_end'] = 'start'
    date_df.loc[date_df['weekday'] > date_df['next_weekday'], 'start_end'] = 'end'

    # Number the weeks, incrementing every start date
    first_start_mask = (date_df['start_end'] == 'start')
    if first_start_mask.any():
        first_start_idx = date_df[first_start_mask].index[0]
    else:
        first_start_idx = 0 # Fallback to first row

    starts_mask = date_df['start_end'] == 'start'
    week_counter = 0
    for idx in date_df[starts_mask].index:
        if idx >= first_start_idx:
            date_df.loc[idx, 'week_num'] = week_counter
            week_counter += 1

    ends_mask = date_df['start_end'] == 'end'
    for idx in date_df[ends_mask].index:
        next_start_idx = date_df[starts_mask & (date_df.index > idx)].index
        if len(next_start_idx) > 0:
            date_df.loc[idx, 'week_num'] = date_df.loc[next_start_idx[0], 'week_num'] - 1
        else:
            date_df.loc[idx, 'week_num'] = week_counter - 1

    # Drop the helper columns
    date_df = date_df.drop(['weekday', 'prev_weekday', 'next_weekday'], axis=1)

    # Merge back to main DataFrame
    df = df.merge(date_df[['date', 'start_end', 'week_num']], on='date', how='left')

    # Drop NaNs in start_end to filter to only beginning and ending dates per trade week
    filtered_df = df.dropna(subset=['start_end']).copy()
    filtered_df = filtered_df.sort_values(['date', 'ticker'], ascending=[True, True]).reset_index(drop=True)

    if filtered_df.empty:
        print("No start/end data found after filtering, returning empty DataFrame")
        return filtered_df

    # Export to .csv, then return the resulting DataFrame for analysis
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved filtered DataFrame to {output_path}")

    return filtered_df

if __name__ == "__main__":
    processed_df = reduce_csv('prices.csv')
    weekly_df = reduce_to_weeks(processed_df)
