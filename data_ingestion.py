import logging
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from prometheus_api_client import PrometheusConnect


logger = logging.getLogger("rightsize.data_ingestion")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DEFAULT_STEP = os.getenv("RIGHTSIZE_PROMETHEUS_STEP", "5m")

prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)


def _step_to_pandas_freq(step: str) -> str:
    value = step.strip().lower()
    if value.endswith("ms"):
        return f"{int(value[:-2])}ms"
    if value.endswith("s"):
        return f"{int(value[:-1])}s"
    if value.endswith("m"):
        return f"{int(value[:-1])}min"
    if value.endswith("h"):
        return f"{int(value[:-1])}h"
    return "5min"


def _series_to_frame(series: dict) -> pd.DataFrame:
    values = series.get("values", [])
    if not values:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame(values, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], unit="s", utc=True).dt.tz_convert(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce") * 1000
    return df.dropna(subset=["ds", "y"])


def _to_naive_utc(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _aggregate_metric_series(
    metric_data: list[dict],
    start_time: datetime,
    end_time: datetime,
    step: str,
) -> pd.DataFrame:
    frames = [_series_to_frame(series) for series in metric_data]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["ds", "y"])

    freq = _step_to_pandas_freq(step)
    df = pd.concat(frames, ignore_index=True)
    df["ds"] = df["ds"].dt.floor(freq)
    df = df.groupby("ds", as_index=False)["y"].sum()
    df = df.sort_values("ds")

    timeline = pd.date_range(
        start=_to_naive_utc(start_time).floor(freq),
        end=_to_naive_utc(end_time).ceil(freq),
        freq=freq,
    )

    df = df.set_index("ds").reindex(timeline)
    df.index.name = "ds"
    df["y"] = df["y"].interpolate(method="time").bfill().ffill()
    df = df.reset_index().dropna(subset=["y"])
    df["y"] = df["y"].clip(lower=0)
    return df


def _query_range(query: str, start_time: datetime, end_time: datetime, step: str) -> list[dict]:
    logger.info("Fetching Prometheus query: %s", query)
    return prom.custom_query_range(
        query=query,
        start_time=start_time,
        end_time=end_time,
        step=step,
    )


def fetch_historical_cpu(
    target_deployment: str,
    days_back: float = 7,
    namespace: str | None = None,
    step: str = DEFAULT_STEP,
) -> pd.DataFrame:
    """
    Fetch CPU usage for all pods belonging to a deployment and return Prophet-ready data.

    Prophet expects:
    - ds: timestamp
    - y: numeric CPU in Kubernetes millicores
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    escaped_name = re.escape(target_deployment)
    namespace_filter = f', namespace="{namespace}"' if namespace else ""

    pod_query = (
        "rate(container_cpu_usage_seconds_total{"
        f'pod=~"{escaped_name}-.*"{namespace_filter}, container!="", image!=""'
        "}[5m])"
    )
    container_fallback_query = (
        "rate(container_cpu_usage_seconds_total{"
        f'container="{target_deployment}"{namespace_filter}, image!=""'
        "}[5m])"
    )

    logger.info(
        "Fetching CPU metrics for deployment=%s namespace=%s from=%s to=%s",
        target_deployment,
        namespace or "*",
        start_time.isoformat(timespec="seconds"),
        end_time.isoformat(timespec="seconds"),
    )

    try:
        metric_data = _query_range(pod_query, start_time, end_time, step)
        if not metric_data:
            logger.info("No pod-regex data found; trying container-name fallback.")
            metric_data = _query_range(container_fallback_query, start_time, end_time, step)
    except Exception as exc:
        logger.warning("Failed to connect to Prometheus: %s", exc)
        return pd.DataFrame(columns=["ds", "y"])

    if not metric_data:
        logger.info("No Prometheus CPU data found for %s.", target_deployment)
        return pd.DataFrame(columns=["ds", "y"])

    df = _aggregate_metric_series(metric_data, start_time, end_time, step)
    logger.info("Prepared %s CPU data points for Prophet.", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- RightSize AI: Data Ingestion Test ---")
    sample = fetch_historical_cpu(target_deployment="prometheus", days_back=0.04)

    if not sample.empty:
        print("\nSuccess! Raw data formatted for ML Engine:")
        print(sample.head())
        print(f"\nTotal data points extracted: {len(sample)}")
    else:
        print("\nNo data returned. Check Prometheus connectivity and target labels.")
