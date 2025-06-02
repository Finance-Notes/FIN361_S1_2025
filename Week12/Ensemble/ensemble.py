import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LassoLars, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load forecasts from each model
df1 = pd.read_csv("outputs/detailed_lasso_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df2 = pd.read_csv("outputs/detailed_ridge_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df3 = pd.read_csv("outputs/detailed_elastic_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df4 = pd.read_csv("outputs/detailed_pcr_1_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df5 = pd.read_csv("outputs/detailed_pcr_2_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df6 = pd.read_csv("outputs/detailed_pcr_3_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df7 = pd.read_csv("outputs/detailed_pcr_4_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df8 = pd.read_csv("outputs/detailed_pcr_5_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df9 = pd.read_csv("outputs/detailed_pcr_6_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df10 = pd.read_csv("outputs/detailed_pcr_7_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])
df11 = pd.read_csv("outputs/detailed_pcr_8_forecasts.csv", usecols=['date', 'stock', 'forecast', 'actual'])

# Rename forecast columns for clarity
df1 = df1.rename(columns={"forecast": "m1_forecast"})
df2 = df2.rename(columns={"forecast": "m2_forecast"})
df3 = df3.rename(columns={"forecast": "m3_forecast"})
df4 = df4.rename(columns={"forecast": "m4_forecast"})
df5 = df5.rename(columns={"forecast": "m5_forecast"})
df6 = df6.rename(columns={"forecast": "m6_forecast"})
df7 = df7.rename(columns={"forecast": "m7_forecast"})
df8 = df8.rename(columns={"forecast": "m8_forecast"})
df9 = df9.rename(columns={"forecast": "m9_forecast"})
df10 = df10.rename(columns={"forecast": "m10_forecast"})
df11 = df11.rename(columns={"forecast": "m11_forecast"})


# Drop 'actual' column from all forecast-only DataFrames (df2 to df11)
for i in range(2, 12):
    vars()['df' + str(i)] = vars()['df' + str(i)].drop(columns=["actual"])

# Merge all forecasts on date and stock
merged = df1
for i in range(2, 12):
    merged = merged.merge(vars()['df' + str(i)], on=["date", "stock"])

# Ensure proper data types
merged["date"] = pd.to_datetime(merged["date"])
merged = merged.sort_values(by=["date", "stock"]).reset_index(drop=True)

# Final columns: date, stock, actual, m1_forecast, m2_forecast, m3_forecast
print(merged.head())

forecast_columns = [f"m{i}_forecast" for i in range(1, 12)]

# Step 1: Create Mean and Median Ensemble Forecasts
merged["mean_forecast"] = merged[forecast_columns].mean(axis=1)
merged["median_forecast"] = merged[forecast_columns].median(axis=1)

# Step 2: Stacked Forecast using OLS
# First 60 observations used for training
train = merged.iloc[:int(len(merged)/24/2)]
test = merged.copy()  # entire dataset for inference

# Features and target
X_train = train[forecast_columns]
y_train = train["actual"]

# Fit OLS model
stacked_model = LinearRegression(fit_intercept=False, positive=True)
stacked_model.fit(X_train, y_train)

"""
# Set up a range of alphas to search over
alphas = np.linspace(0, 0.00001, 1000)

# Use GridSearchCV to find the alpha that results in exactly 5 non-zero coefficients
best_model = None
for alpha in alphas:
    model = LassoLars(alpha=alpha, fit_intercept=False, positive=True, max_iter=10000)
    model.fit(X_train, y_train)
    nonzero_count = np.sum(model.coef_ != 0)
    print(nonzero_count)
    if nonzero_count == 4:
        best_model = model
        break

if best_model is None:
    raise ValueError("No alpha found that results in exactly 5 non-zero coefficients.")

stacked_model = best_model


# Step 2: Normalize non-zero coefficients to sum to 1
coefs = best_model.coef_.copy()
mask = coefs != 0
coefs[mask] = coefs[mask] / coefs[mask].sum()
"""

"""
# Fit LassoCV model with 3-fold cross-validation
lasso_cv = LassoCV(cv=5, fit_intercept=False, positive=True, max_iter=10000, random_state=0)
lasso_cv.fit(X_train, y_train)

# Extract best model
stacked_model = lasso_cv

# Get number of non-zero coefficients
nonzero_count = (stacked_model.coef_ != 0).sum()

print(f"Selected alpha: {stacked_model.alpha_}")
print(f"Non-zero coefficients: {nonzero_count}")
"""

# Predict stacked forecasts over full sample
X_all = merged[forecast_columns]
merged["stacked_forecast"] = stacked_model.predict(X_all)

#merged["stacked_forecast"] = X_all.values @ coefs

# Define evaluation subset: out-of-sample (after first 60 observations)
oos = merged.iloc[int(len(merged)/24/2):].copy()
y_true = oos["actual"]

# Define forecast columns to evaluate
forecast_cols = forecast_columns + ["mean_forecast", "median_forecast", "stacked_forecast"]

# Storage for results
results = []

for col in forecast_cols:
    y_pred = oos[col]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    results.append({
        "Model": col,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": direction_accuracy
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by="RMSE")
print(results_df)

