import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_DATA_FILE = "inr_macro_monthly.csv"
STATIONARITY_RESULTS = "stationarity_tests.csv"
VAR_RESULTS_FILE = "var_model_summary.txt"
COINTEGRATION_FILE = "cointegration_results.txt"
FORECAST_FILE = "24month_forecast.csv"
IRF_FILE = "impulse_responses.csv"
TREND_PLOT_FILE = "macro_trends.png"
IRF_PLOT_FILE = "impulse_response_plots.png"
RESULTS_DIR = "analysis_outputs"

def generate_macro_data():
    dates = pd.date_range("2010-01-01", "2023-12-01", freq='M')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'inr_usd': np.cumsum(np.random.normal(0.002, 0.15, len(dates))) + 45,
        'inflation_diff': np.random.normal(2.5, 1.2, len(dates)),
        'interest_diff': np.random.normal(-1.8, 0.9, len(dates)),
        'current_account': np.random.normal(-2.3, 1.5, len(dates)),
        'fii_flows': np.cumsum(np.random.normal(0, 500, len(dates))),
        'crude_prices': np.cumsum(np.random.normal(0.1, 3, len(dates))) + 60
    }).set_index('date')
    
    data.to_csv(INPUT_DATA_FILE)
    return data

def check_stationarity(df):
    results = []
    for col in df.columns:
        adf_result = adfuller(df[col])
        results.append({
            'variable': col,
            'test_statistic': adf_result[0],
            'p_value': adf_result[1],
            '1pct_critical': adf_result[4]['1%'],
            '5pct_critical': adf_result[4]['5%']
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(STATIONARITY_RESULTS, index=False)
    return results_df

def build_var_model(data_diff):
    model = VAR(data_diff)
    var_result = model.fit(4)
    
    with open(VAR_RESULTS_FILE, 'w') as f:
        f.write(str(var_result.summary()))
    
    return var_result

def test_cointegration(df):
    coint_results = []
    for col in df.columns[1:]:
        score, pval, _ = coint(df['inr_usd'], df[col])
        coint_results.append({
            'variable_pair': f"INR_USD ~ {col}",
            't_statistic': score,
            'p_value': pval
        })
    
    coint_df = pd.DataFrame(coint_results)
    coint_df.to_csv(COINTEGRATION_FILE, index=False)
    return coint_df

def generate_var_forecasts(model):
    forecast = model.forecast(model.y[-model.p:], 24)
    forecast_df = pd.DataFrame(forecast, 
                  index=pd.date_range(model.data.endog.index[-1], 
                                    periods=25, freq='M')[1:],
                  columns=model.names)
    
    forecast_df.to_csv(FORECAST_FILE)
    return forecast_df

def calculate_impulse_responses(model):
    irf = model.irf(periods=12)
    irf_df = pd.DataFrame(irf.irfs, columns=[f"Shock_{x}" for x in model.names])
    irf_df.to_csv(IRF_FILE)
    
    plt.figure(figsize=(12,8))
    irf.plot(orth=False)
    plt.savefig(IRF_PLOT_FILE)
    plt.close()

def create_trend_visualization(df):
    plt.figure(figsize=(10,12))
    
    plt.subplot(3,1,1)
    df['inr_usd'].plot(color='navy')
    plt.title('INR/USD Exchange Rate Trend')
    plt.ylabel('Rupees per USD')
    
    plt.subplot(3,1,2)
    df[['inflation_diff','interest_diff']].plot()
    plt.title('Inflation and Interest Rate Differentials')
    
    plt.subplot(3,1,3)
    df['crude_prices'].plot(color='red')
    plt.title('Crude Oil Price Trend')
    
    plt.tight_layout()
    plt.savefig(TREND_PLOT_FILE)
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Running Rupee Depreciation Analysis")
    macro_data = generate_macro_data()
    
    print("Conducting stationarity tests")
    stationarity_results = check_stationarity(macro_data)
    
    print("Building VAR model")
    diff_data = macro_data.diff().dropna()
    var_model = build_var_model(diff_data)
    
    print("Testing cointegration relationships")
    coint_results = test_cointegration(macro_data)
    
    print("Generating forecasts")
    forecasts = generate_var_forecasts(var_model)
    
    print("Calculating impulse responses")
    calculate_impulse_responses(var_model)
    
    print("Creating visualization plots")
    create_trend_visualization(macro_data)
    
    print("Analysis complete. Output files generated:")
    print(f"Input data: {INPUT_DATA_FILE}")
    print(f"Stationarity tests: {STATIONARITY_RESULTS}")
    print(f"VAR model: {VAR_RESULTS_FILE}")
    print(f"Cointegration results: {COINTEGRATION_FILE}")
    print(f"Forecasts: {FORECAST_FILE}")
    print(f"Impulse responses: {IRF_FILE}")
    print(f"Trend plots: {TREND_PLOT_FILE}")
    print(f"IRF plots: {IRF_PLOT_FILE}")

if __name__ == "__main__":
    main()
