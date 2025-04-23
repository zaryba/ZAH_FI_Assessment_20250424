import pandas as pd
import numpy as np
import datetime as dt
import logging


class IndexModel:
    def __init__(self, initial_level=100, top_n=3, weights=None):
        self.index_levels = None
        self.initial_level = initial_level
        self.top_n = top_n
        self.weights = weights if weights else [0.5, 0.25, 0.25]
        logging.basicConfig(level=logging.INFO)

    def calc_index_level(self, start_date: dt.date, end_date: dt.date):
        logging.info("Loading stock prices...")
        stock_prices = pd.read_csv('data_sources/stock_prices.csv', parse_dates=['Date'], dayfirst=True)
        stock_prices = stock_prices.set_index('Date')

        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        stock_prices = stock_prices.reindex(business_days).fillna(method='bfill')

        index_level = self.initial_level
        index_levels = []

        for month_start in pd.date_range(start=start_date, end=end_date, freq='MS'):
            # Get the last business day of the previous month
            prev_month_end = (month_start - pd.Timedelta(days=1))
            if prev_month_end not in stock_prices.index:
                # Find the last available date before prev_month_end
                valid_dates = stock_prices.index[stock_prices.index < prev_month_end]
                if valid_dates.empty:
                    # Skip the month if no valid date exists
                    continue
                prev_month_end = valid_dates[-1]

            # Calculate market capitalization
            market_caps = stock_prices.loc[prev_month_end]

            # Select top 3 stocks
            top_stocks = market_caps.nlargest(3).index
            weights = [0.5, 0.25, 0.25]

            # Calculate daily index levels for the month
            for day in pd.date_range(start=month_start, end=month_start + pd.offsets.MonthEnd(0), freq='B'):
                if day not in stock_prices.index:
                    continue
                daily_prices = stock_prices.loc[day, top_stocks]
                daily_return = np.dot(daily_prices / stock_prices.loc[prev_month_end, top_stocks], weights)
                index_level *= daily_return
                index_levels.append({'Date': day, 'Index Level': index_level})
        self.index_levels = pd.DataFrame(index_levels)
        logging.info("Index calculation completed.")

    def export_values(self, output_file: str):
        if self.index_levels is not None:
            self.index_levels.to_csv(output_file, index=False)
            logging.info(f"Index levels exported to {output_file}.")
        else:
            raise ValueError("Index levels have not been calculated yet.")
