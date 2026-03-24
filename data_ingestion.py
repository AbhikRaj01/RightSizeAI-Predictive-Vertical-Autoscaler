from prometheus_api_client import PrometheusConnect
import pandas as pd
from datetime import datetime, timedelta

# Connect to Prometheus (Assuming it's port-forwarded to localhost:9090)
prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)

def fetch_historical_cpu(target_deployment: str, days_back: int = 7) -> pd.DataFrame:
    """
    Fetches raw CPU usage from Prometheus and formats it for the Prophet ML model.
    """
    # The PromQL Query: Calculates per-second CPU usage over a 5-minute rolling window
    query = f'sum(rate(container_cpu_usage_seconds_total{{container="{target_deployment}"}}[5m]))'
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    print(f"Fetching metrics for '{target_deployment}' from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}...")
    
    try:
        metric_data = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step="5m"
        )
    except Exception as e:
        print(f"Failed to connect to Prometheus: {e}")
        return pd.DataFrame()

    if not metric_data:
        print("No data found. Is Prometheus running and is the container active?")
        return pd.DataFrame()

    # Process into a Pandas DataFrame (Prophet requires 'ds' for date and 'y' for value)
    raw_values = metric_data[0]['values']
    df = pd.DataFrame(raw_values, columns=['ds', 'y'])
    
    # Clean up the data types
    df['ds'] = pd.to_datetime(df['ds'], unit='s')
    df['y'] = df['y'].astype(float) * 1000  # Convert to Kubernetes millicores (m)
    
    return df

if __name__ == "__main__":
    # Test execution
    print("--- RightSize AI: Data Ingestion Test ---")
    # Look at Prometheus, and only ask for the last hour (0.04 days)
    df = fetch_historical_cpu(target_deployment="prometheus", days_back=0.04)
    
    if not df.empty:
        print("\nSuccess! Raw data formatted for ML Engine:")
        print(df.head())
        print(f"\nTotal data points extracted: {len(df)}")