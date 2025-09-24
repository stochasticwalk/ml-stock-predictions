# ml-stock-predictions
A machine learning approach to the stock market

**Note: This project or its outputs should not serve as financial advice. Do your own research on investments.**

## Core Theory

This project works under the assumption that the stock market is, in some ways, reactive to itself. At least in the short term. It assumes the overall performance of the stock market has some level of predictive power to the performance of individual stocks in the near future.

Please note that the data I used for this project came from the EOD product of [data.nasdaq.com](https://data.nasdaq.com). Seeing as it is a paid service I will not be providing the inputs or outputs of this project. Please find or purchase your own data and adjust the files as necessary if you would like to try this out for yourself.

## Order of operations

1. Run `preproc.py`, importing your stock `.csv` file. This project assumes you have consistent stock data back to at least `2012-01-01`, adjust as necessary if otherwise. Column names and types are listed in the inital function of this file, adjust them as necessary for your particular dataset.
2. Run `train.py`, importing the `.csv` file you got as the output of `preproc.py`. You will get a `/models` folder in your project location with a model per-stock saved as a `.joblib` file, as well as a `.csv` file listing the accuracy of each ticker's model (Range [-0.5, 1.0], higher is better, by default models with an accuracy score <0.5 will not be saved) and a `.json` file containing a list of stock tickers used for training (needed for future predictions).
3. To make future predictions, download the first day of a given trading week (expected filename is `week_start.csv`) and the last day of that week (expected filename is `week_end.csv`) and run `predictions.py`. You will get a resulting `.csv` file that gives each ticker an integer label from 1-7, predicting stock performance from low to high (i.e., 7 is predicted to have a very positive week, 1 is predicted to have a very negative week)
