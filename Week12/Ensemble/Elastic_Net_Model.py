"""
DJIA Stock Forecasting Data Pipeline

This module provides functionality to download, process, and combine Dow Jones Industrial Average
stock data with Federal Reserve Economic Data (FRED-MD) for forecasting applications.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
from pathlib import Path
from sklearn.linear_model import LassoLars, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# Constants
DOW_TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS',
    'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
    'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'DOW'
]

FRED_MD_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64"
)

STOCK_DATA_FILE = 'dow_tickers.csv'
LOOKBACK_PERIOD = '40y'
FREQUENCY = '1mo'


class StockDataProcessor:
    """Handles downloading and processing of stock price data."""

    def __init__(self, tickers: List[str] = DOW_TICKERS):
        """
        Initialize the stock data processor.

        Args:
            tickers: List of stock ticker symbols to process
        """
        self.tickers = tickers

    def download_stock_data(self, save_to_file: bool = True) -> pd.DataFrame:
        """
        Download adjusted closing prices for specified tickers.

        Args:
            save_to_file: Whether to save data to CSV file

        Returns:
            DataFrame with adjusted closing prices

        Raises:
            ValueError: If no data is successfully downloaded
        """
        try:
            print(f"Downloading stock data for {len(self.tickers)} tickers...")
            data = yf.download(
                self.tickers,
                interval=FREQUENCY,
                period=LOOKBACK_PERIOD,
                progress=False
            )['Close']

            if data.empty:
                raise ValueError("No stock data downloaded")

            # Remove tickers with insufficient data
            data = data.dropna(axis=1)

            if save_to_file:
                data.to_csv(STOCK_DATA_FILE)
                print(f"Stock data saved to {STOCK_DATA_FILE}")

            print(f"Successfully downloaded data for {len(data.columns)} stocks")
            return data

        except Exception as e:
            print(f"Error downloading stock data: {e}")
            return self._load_backup_data()

    def _load_backup_data(self) -> pd.DataFrame:
        """Load data from backup CSV file."""
        try:
            print(f"Loading backup data from {STOCK_DATA_FILE}")
            data = pd.read_csv(STOCK_DATA_FILE, index_col=0, parse_dates=True)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not download data and backup file {STOCK_DATA_FILE} not found"
            )

    def calculate_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from price data.

        Args:
            prices: DataFrame with price data

        Returns:
            DataFrame with log returns
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns


class FredDataProcessor:
    """Handles downloading and processing of FRED-MD economic data."""

    def __init__(self, fred_url: str = FRED_MD_URL):
        """
        Initialize the FRED data processor.

        Args:
            fred_url: URL to FRED-MD dataset
        """
        self.fred_url = fred_url

    def download_fred_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download and process FRED-MD dataset.

        Returns:
            Tuple of (processed_data, transformation_codes)
        """
        print("Downloading FRED-MD data (this may take 1-2 minutes)...")

        try:
            # Download data
            raw_data = pd.read_csv(self.fred_url)

            # Remove last 3 rows (usually contain metadata)
            raw_data = raw_data.iloc[:-3]

            # Extract transformation codes from first row
            transformation_codes = raw_data.iloc[0, 1:]

            # Remove transformation codes row
            raw_data = raw_data.drop(raw_data.index[0])

            # Set date index
            raw_data = raw_data.set_index("sasdate", drop=True)
            raw_data.index = pd.to_datetime(raw_data.index, format='%m/%d/%Y')

            # Convert to numeric, replacing non-numeric values with NaN
            for col in raw_data.columns:
                raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

            print(f"Successfully downloaded FRED-MD data: {raw_data.shape}")
            return raw_data, transformation_codes

        except Exception as e:
            raise ConnectionError(f"Failed to download FRED-MD data: {e}")

    def transform_series(self, series: pd.Series, transformation_code: int) -> pd.Series:
        """
        Transform a time series according to FRED-MD transformation codes.

        Args:
            series: Time series to transform
            transformation_code: Transformation code (1-7)

        Returns:
            Transformed series

        Transformation codes:
            1: No transformation
            2: First difference
            3: Second difference  
            4: Natural log
            5: First difference of natural log
            6: Second difference of natural log
            7: First difference of percent change
        """
        series = series.copy()
        small_value = 1e-6

        if transformation_code == 1:  # No transformation
            return series

        elif transformation_code == 2:  # First difference
            return series.diff()

        elif transformation_code == 3:  # Second difference
            return series.diff().diff()

        elif transformation_code == 4:  # Natural log
            if (series <= small_value).any():
                warnings.warn("Series contains non-positive values, returning NaN")
                return pd.Series(np.nan, index=series.index)
            return np.log(series)

        elif transformation_code == 5:  # First difference of natural log
            if (series <= small_value).any():
                warnings.warn("Series contains non-positive values, returning NaN")
                return pd.Series(np.nan, index=series.index)
            return np.log(series).diff()

        elif transformation_code == 6:  # Second difference of natural log
            if (series <= small_value).any():
                warnings.warn("Series contains non-positive values, returning NaN")
                return pd.Series(np.nan, index=series.index)
            return np.log(series).diff().diff()

        elif transformation_code == 7:  # First difference of percent change
            pct_change = series.pct_change()
            return pct_change.diff()

        else:
            raise ValueError(f"Invalid transformation code: {transformation_code}")

    def apply_transformations(self, data: pd.DataFrame,
                              transformation_codes: pd.Series) -> pd.DataFrame:
        """
        Apply transformations to all series in the dataset.

        Args:
            data: Raw FRED-MD data
            transformation_codes: Transformation codes for each series

        Returns:
            DataFrame with transformed series
        """
        transformed_data = pd.DataFrame(index=data.index)

        for column in data.columns:
            if column in transformation_codes.index:
                tcode = int(transformation_codes[column])
                transformed_series = self.transform_series(data[column], tcode)
                transformed_data[column] = transformed_series

        return transformed_data


class DataCombiner:
    """Combines stock and economic data on aligned datetime index."""

    @staticmethod
    def align_and_combine(stock_returns: pd.DataFrame,
                          economic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align and combine stock returns with economic data.

        Args:
            stock_returns: Stock log returns data
            economic_data: Transformed economic indicator data

        Returns:
            Combined dataset with aligned datetime index
        """
        print("Aligning and combining datasets...")

        # Ensure both indices are datetime
        stock_returns.index = pd.to_datetime(stock_returns.index)
        economic_data.index = pd.to_datetime(economic_data.index)

        # Find overlapping date range
        start_date = max(stock_returns.index.min(), economic_data.index.min())
        end_date = min(stock_returns.index.max(), economic_data.index.max())

        print(f"Data overlap period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

        # Align on monthly frequency and combine
        combined_data = pd.concat([
            stock_returns.loc[start_date:end_date],
            economic_data.loc[start_date:end_date]
        ], axis=1, join='inner')

        # Add prefix to distinguish data types
        stock_cols = stock_returns.columns
        econ_cols = economic_data.columns

        # Rename columns for clarity
        renamed_cols = {}
        for col in combined_data.columns:
            if col in stock_cols:
                renamed_cols[col] = f"stock_{col}"
            elif col in econ_cols:
                renamed_cols[col] = f"econ_{col}"

        combined_data = combined_data.rename(columns=renamed_cols)

        print(f"Combined dataset shape: {combined_data.shape}")
        print(f"Stock variables: {len(stock_cols)}")
        print(f"Economic variables: {len(econ_cols)}")

        return combined_data


def main() -> pd.DataFrame:
    """
    Main execution function to download, process, and combine all data.

    Returns:
        Combined dataset ready for forecasting
    """
    # Initialize processors
    stock_processor = StockDataProcessor()
    fred_processor = FredDataProcessor()

    # Download and process stock data
    stock_prices = stock_processor.download_stock_data()
    stock_returns = stock_processor.calculate_log_returns(stock_prices)

    # Download and process FRED data
    fred_raw_data, transformation_codes = fred_processor.download_fred_data()
    fred_transformed = fred_processor.apply_transformations(fred_raw_data, transformation_codes)

    # Combine datasets
    combined_dataset = DataCombiner.align_and_combine(stock_returns, fred_transformed)
    combined_dataset = combined_dataset.dropna(axis=1)

    # Display summary statistics
    print("\n" + "=" * 50)
    print("DATA PROCESSING COMPLETE")
    print("=" * 50)
    print(
        f"Final dataset period: {combined_dataset.index[0].strftime('%Y-%m')} to {combined_dataset.index[-1].strftime('%Y-%m')}")
    print(f"Total observations: {len(combined_dataset)}")
    print(f"Total variables: {len(combined_dataset.columns)}")
    print(f"Missing values: {combined_dataset.isnull().sum().sum()}")

    return combined_dataset


class LassoForecastingEngine:
    """
    Implements expanding window Lasso forecasting with custom validation.

    This class performs one-month ahead stock return forecasting using economic
    indicators as predictors with Lasso regularization estimated via LARS.
    """

    def __init__(self, validation_size: int = 144, initial_training_size: int = 216):
        """
        Initialize the forecasting engine.

        Args:
            validation_size: Number of observations in validation set
            initial_training_size: Number of observations in initial training set
        """
        self.validation_size = validation_size
        self.initial_training_size = initial_training_size
        self.min_window_size = validation_size + initial_training_size

        # Store results
        self.forecasts = {}
        self.actuals = {}
        self.optimal_alphas = {}
        self.feature_importance = {}

    def prepare_features_and_targets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare lagged features and forward-looking targets for forecasting.

        Args:
            data: Combined dataset with stock returns and economic indicators

        Returns:
            Tuple of (features_lagged, targets_forward)
        """
        # Separate stock returns and economic indicators
        stock_cols = [col for col in data.columns if col.startswith('stock_')]
        econ_cols = [col for col in data.columns if col.startswith('econ_')]

        print(f"Identified {len(stock_cols)} stock variables and {len(econ_cols)} economic variables")

        # Economic indicators at time t (lagged features)
        features = data[econ_cols].copy()

        # Stock returns at time t+1 (forward targets)
        targets = data[stock_cols].shift(-1).copy()  # Shift backward to get future returns

        # Remove rows with missing values
        valid_data = pd.concat([features, targets], axis=1).dropna()

        features_clean = valid_data[econ_cols]
        targets_clean = valid_data[stock_cols]

        print(f"Data prepared: {len(features_clean)} observations after cleaning")

        return features_clean, targets_clean

    def custom_lasso_validation(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                alpha_grid: np.ndarray,
                                beta_grid: np.ndarray
                                ):
        """
        Custom validation function to find optimal Lasso penalty parameter.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features  
            y_val: Validation targets
            alpha_grid: Grid of penalty parameters to test

        Returns:
            Tuple of (optimal_alpha, validation_mse_scores)
        """
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        validation_scores = []
        param_grid =[]

        for alpha in alpha_grid:
            for beta in beta_grid:
                # Fit ElasticNet (Lasso with LARS equivalent when l1_ratio=1)
                lasso = ElasticNet(alpha=alpha, l1_ratio=beta, max_iter=1000)
                lasso.fit(X_train_scaled, y_train)

                # Predict on validation set
                y_val_pred = lasso.predict(X_val_scaled)

                # Calculate MSE and store parameters
                mse = mean_squared_error(y_val, y_val_pred)
                validation_scores.append(mse)
                param_grid.append((alpha, beta))

        # Find optimal parameters
        optimal_idx = np.argmin(validation_scores)
        optimal_alpha, optimal_l1 = param_grid[optimal_idx]

        return optimal_alpha, optimal_l1, np.array(validation_scores)

    def forecast_single_stock(self, features: pd.DataFrame, targets: pd.Series,
                              stock_name: str) -> dict:
        """
        Perform expanding window forecasting for a single stock.

        Args:
            features: Economic indicator features
            targets: Stock return targets
            stock_name: Name of the stock being forecasted

        Returns:
            Dictionary with forecasting results
        """
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler

        n_obs = len(features)
        if n_obs < self.min_window_size:
            raise ValueError(f"Insufficient data: {n_obs} < {self.min_window_size}")

        # Initialize storage
        forecasts = []
        actuals = []
        forecast_dates = []
        optimal_alphas = []

        # Define alpha grid for validation
        alpha_grid = np.logspace(-4, 1, 20)  # From 0.0001 to 10
        beta_grid = np.linspace(0.0001, 1, 4)  # From 0.0001 to 10

        print(f"Forecasting {stock_name}...")

        # Expanding window loop
        for window_end in range(self.min_window_size, n_obs):
            # Define current window
            window_start = 0

            # Split into training and validation within current window
            train_end = window_end - self.validation_size
            val_start = train_end
            val_end = window_end

            # Extract training data
            X_train = features.iloc[window_start:train_end].values
            y_train = targets.iloc[window_start:train_end].values

            # Extract validation data
            X_val = features.iloc[val_start:val_end].values
            y_val = targets.iloc[val_start:val_end].values

            # Skip if insufficient data
            if len(X_train) < 50 or len(X_val) < 10:
                continue

            # Find optimal alpha using custom validation
            try:
                optimal_alpha, optimal_l1, _ = self.custom_lasso_validation(
                    X_train, y_train, X_val, y_val, alpha_grid, beta_grid)
                optimal_alphas.append(optimal_alpha)

                # Retrain on full training + validation data with optimal alpha
                X_full_train = features.iloc[window_start:val_end].values
                y_full_train = targets.iloc[window_start:val_end].values

                # Standardize features
                scaler = StandardScaler()
                X_full_train_scaled = scaler.fit_transform(X_full_train)

                # Fit final model
                final_model = ElasticNet(alpha=optimal_alpha, l1_ratio=optimal_l1, max_iter=1000)
                final_model.fit(X_full_train_scaled, y_full_train)

                # Make out-of-sample forecast (one step ahead)
                if window_end < n_obs:
                    X_forecast = features.iloc[window_end:window_end + 1].values
                    X_forecast_scaled = scaler.transform(X_forecast)

                    forecast = final_model.predict(X_forecast_scaled)[0]
                    actual = targets.iloc[window_end]

                    forecasts.append(forecast)
                    actuals.append(actual)
                    forecast_dates.append(features.index[window_end])

            except Exception as e:
                print(f"  Warning: Skipping window ending at {window_end}: {e}")
                continue

        results = {
            'forecasts': np.array(forecasts),
            'actuals': np.array(actuals),
            'dates': forecast_dates,
            'optimal_alphas': np.array(optimal_alphas),
            'n_forecasts': len(forecasts)
        }

        print(f"  Generated {len(forecasts)} forecasts for {stock_name}")

        return results

    def run_forecasting(self, data: pd.DataFrame) -> dict:
        """
        Run expanding window Lasso forecasting for all stocks.

        Args:
            data: Combined dataset

        Returns:
            Dictionary with all forecasting results
        """
        print("\n" + "=" * 60)
        print("STARTING LASSO FORECASTING WITH EXPANDING WINDOWS")
        print("=" * 60)

        # Prepare features and targets
        features, targets = self.prepare_features_and_targets(data)

        # Get stock names
        stock_names = [col.replace('stock_', '') for col in targets.columns]

        print(f"Forecasting setup:")
        print(f"  - Initial training size: {self.initial_training_size}")
        print(f"  - Validation size: {self.validation_size}")
        print(f"  - Minimum window size: {self.min_window_size}")
        print(f"  - Available observations: {len(features)}")
        print(f"  - Expected forecasts per stock: ~{len(features) - self.min_window_size}")
        print(f"  - Number of stocks: {len(stock_names)}")

        # Run forecasting for each stock
        all_results = {}

        for i, stock_name in enumerate(stock_names, 1):
            print(f"\n[{i}/{len(stock_names)}] Processing {stock_name}")

            try:
                target_series = targets[f'stock_{stock_name}']
                results = self.forecast_single_stock(features, target_series, stock_name)
                all_results[stock_name] = results

            except Exception as e:
                print(f"  Error forecasting {stock_name}: {e}")
                continue

        print("\n" + "=" * 60)
        print("FORECASTING COMPLETE")
        print("=" * 60)

        # Summary statistics
        total_forecasts = sum(results['n_forecasts'] for results in all_results.values())
        successful_stocks = len(all_results)

        print(f"Summary:")
        print(f"  - Successfully forecasted stocks: {successful_stocks}/{len(stock_names)}")
        print(f"  - Total forecasts generated: {total_forecasts}")
        print(f"  - Average forecasts per stock: {total_forecasts / max(successful_stocks, 1):.1f}")

        return all_results

    def save_detailed_forecasts(self, results: dict, filename: str = 'outputs/detailed_elastic_forecasts.csv') -> pd.DataFrame:
        """
        Save detailed forecast and actual returns for each stock and period to CSV.

        Args:
            results: Dictionary with forecasting results
            filename: Name of output CSV file

        Returns:
            DataFrame with detailed forecast results
        """
        detailed_data = []

        for stock_name, stock_results in results.items():
            forecasts = stock_results['forecasts']
            actuals = stock_results['actuals']
            dates = stock_results['dates']

            for i in range(len(forecasts)):
                forecast_error = actuals[i] - forecasts[i]

                detailed_data.append({
                    'date': dates[i],
                    'stock': stock_name,
                    'forecast': forecasts[i],
                    'actual': actuals[i],
                    'forecast_error': forecast_error,
                    'absolute_error': abs(forecast_error),
                    'squared_error': forecast_error ** 2
                })

        # Create DataFrame
        detailed_df = pd.DataFrame(detailed_data)

        if not detailed_df.empty:
            # Sort by date and stock
            detailed_df = detailed_df.sort_values(['date', 'stock'])

            # Save to CSV
            detailed_df.to_csv(filename, index=False)
            print(f"Detailed forecasts saved to {filename}")

            # Display summary
            print(f"\nDetailed forecast data summary:")
            print(f"  - Total forecast-actual pairs: {len(detailed_df)}")
            print(
                f"  - Date range: {detailed_df['date'].min().strftime('%Y-%m')} to {detailed_df['date'].max().strftime('%Y-%m')}")
            print(f"  - Stocks covered: {detailed_df['stock'].nunique()}")

            # Show sample of the data
            print(f"\nSample of detailed forecast data:")
            print(detailed_df.head(10).to_string(index=False, float_format='%.4f'))

        return detailed_df

    def evaluate_forecasts(self, results: dict) -> pd.DataFrame:
        """
        Evaluate forecasting performance across all stocks.

        Args:
            results: Dictionary with forecasting results

        Returns:
            DataFrame with performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        performance_metrics = []

        for stock_name, stock_results in results.items():
            forecasts = stock_results['forecasts']
            actuals = stock_results['actuals']

            if len(forecasts) > 0:
                mse = mean_squared_error(actuals, forecasts)
                mae = mean_absolute_error(actuals, forecasts)
                rmse = np.sqrt(mse)

                # Additional metrics
                correlation = np.corrcoef(actuals, forecasts)[0, 1] if len(forecasts) > 1 else np.nan
                directional_accuracy = np.mean(np.sign(actuals) == np.sign(forecasts))

                # Mean optimal alpha
                mean_alpha = np.mean(stock_results['optimal_alphas'])

                performance_metrics.append({
                    'stock': stock_name,
                    'n_forecasts': len(forecasts),
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'correlation': correlation,
                    'directional_accuracy': directional_accuracy,
                    'mean_optimal_alpha': mean_alpha
                })

        performance_df = pd.DataFrame(performance_metrics)

        if not performance_df.empty:
            # Sort by MSE
            performance_df = performance_df.sort_values('correlation', ascending=False)

            print("\nFORECASTING PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f"{'Stock':<6} {'N':<4} {'RMSE':<8} {'MAE':<8} {'Corr':<6} {'Dir.Acc':<8} {'Alpha':<8}")
            print("-" * 50)

            for _, row in performance_df.head(10).iterrows():
                print(f"{row['stock']:<6} {row['n_forecasts']:<4.0f} "
                      f"{row['rmse']:<8.4f} {row['mae']:<8.4f} "
                      f"{row['correlation']:<6.3f} {row['directional_accuracy']:<8.3f} "
                      f"{row['mean_optimal_alpha']:<8.4f}")

        return performance_df


def run_complete_forecasting_pipeline():
    """Execute the complete data processing and forecasting pipeline."""

    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    forecasting_dataset = main()

    # Step 2: Initialize and run forecasting
    print("\nStep 2: Initializing forecasting engine...")
    forecasting_engine = LassoForecastingEngine(
        validation_size=12,
        initial_training_size=120
    )

    # Step 3: Generate forecasts
    forecast_results = forecasting_engine.run_forecasting(forecasting_dataset)

    # Step 4: Evaluate performance
    print("\nStep 4: Evaluating forecasting performance...")
    performance_metrics = forecasting_engine.evaluate_forecasts(forecast_results)

    # Step 5: Save detailed forecasts
    print("\nStep 5: Saving detailed forecast results...")
    detailed_forecasts = forecasting_engine.save_detailed_forecasts(forecast_results)

    # Step 6: Save results
    forecasting_dataset.to_csv('outputs/djia_forecasting_dataset.csv')
    performance_metrics.to_csv('outputs/forecasting_performance.csv', index=False)

    print(f"\nResults saved:")
    print(f"  - Dataset: djia_forecasting_dataset.csv")
    print(f"  - Performance: forecasting_performance.csv")
    print(f"  - Detailed forecasts: detailed_forecasts.csv")

    return forecasting_dataset, forecast_results, performance_metrics, detailed_forecasts


if __name__ == '__main__':
    # Execute the complete forecasting pipeline
    dataset, results, performance, detailed_forecasts = run_complete_forecasting_pipeline()
