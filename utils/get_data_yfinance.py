"""
Get financial data using yFinance API.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
TICKERS = {
    'AAPL': 'Apple Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'SPY': 'S&P 500 ETF',  
    'BTC-USD': 'Bitcoin USD'  
}

OUTPUT_DIR = '../data_new/'
START_DATE = '2000-01-01'  
END_DATE = None  
INTERVAL = '1d'

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def download_data(ticker, name, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL):
    """
    Download data for a given ticker with customizable parameters.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol (e.g., 'AAPL', 'BTC-USD')
    name : str
        The full name of the asset
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'. Default: '2000-01-01'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'. Default: None (today)
    interval : str, optional
        Data interval: '1d', '1h', '5m', etc. Default: '1d'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the historical data
    """
    print(f"\n{'='*70}")
    print(f"Downloading: {ticker} - {name}")
    print(f"{'='*70}")
    
    try:
        # Create ticker object
        asset = yf.Ticker(ticker)
        
        # Download data
        print(f"Fetching data (from: {start_date}, to: {end_date or 'today'}, interval: {interval})...")
        df = asset.history(start=start_date, end=end_date, interval=interval)
        
        # If no data with start_date, try period="max"
        if df.empty:
            print(f"  Trying with period='max'...")
            df = asset.history(period="max", interval=interval)
        
        if df.empty:
            print(f"[ERROR] No data available for {ticker}")
            return None
        
        # Calculate years of data
        years = (df.index.max() - df.index.min()).days / 365.25
        
        # Data information
        print(f"[OK] {len(df)} data points retrieved")
        print(f"  Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"  Years covered: {years:.1f} years")
        print(f"  Calendar days: {(df.index.max() - df.index.min()).days}")
        
        # Filename (replace - with _ for consistency)
        interval_str = interval.replace('h', 'hour').replace('d', 'daily').replace('m', 'min')
        filename = f"{ticker.replace('-', '_')}_{interval_str}_yfinance.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save
        df.to_csv(filepath)
        print(f"[OK] Saved: {filepath}")
        
        # Display preview
        print(f"\nData preview:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Error downloading {ticker}: {e}")
        return None


def main(start_date=START_DATE, end_date=END_DATE, interval=INTERVAL):
    """
    Download all assets with customizable parameters
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'. Default: '2000-01-01'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'. Default: None (today)
    interval : str, optional
        Data interval: '1d', '1h', '5m', etc. Default: '1d'
    """
    print("\n" + "="*70)
    print("DATA DOWNLOAD - YFINANCE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Period: {start_date} to {end_date or 'today'}")
    print(f"Interval: {interval}")
    print(f"Number of assets: {len(TICKERS)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Download each asset
    results = {}
    for ticker, name in TICKERS.items():
        df = download_data(ticker, name, start_date=start_date, end_date=end_date, interval=interval)
        if df is not None:
            results[ticker] = df
    
    # Final summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    if results:
        summary_data = []
        for ticker, df in results.items():
            years = (df.index.max() - df.index.min()).days / 365.25
            summary_data.append({
                'Ticker': ticker,
                'Name': TICKERS[ticker],
                'Data Points': len(df),
                'Start': df.index.min().strftime('%Y-%m-%d'),
                'End': df.index.max().strftime('%Y-%m-%d'),
                'Years': f"{years:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        print(f"\n[OK] {len(results)}/{len(TICKERS)} assets downloaded successfully")
        print(f"[OK] Files saved in: {OUTPUT_DIR}")
        
        # Note about data
        print(f"\nNote:")
        print(f"  - Stocks (AAPL, AMZN, NVDA, SPY): ~25 years of data (depending on IPO)")
        print(f"  - BTC-USD: Data since ~2014 (limited crypto availability)")
    else:
        print("\n[ERROR] No assets downloaded")
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
