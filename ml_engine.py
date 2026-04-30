import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prophet import Prophet

from data_ingestion import fetch_historical_cpu
from scaling_history import (
    log_operator_decision,
    log_prediction,
    log_scaling_action,
    read_operator_decisions,
    read_prediction_history,
    read_scaling_history,
    upsert_deployment_config,
)

try:
    import kubernetes.client as k8s
    from kubernetes import config as k8s_config
except Exception:  # pragma: no cover - lets dashboard demo run without Kubernetes installed.
    k8s = None
    k8s_config = None


logging.basicConfig(
    level=os.getenv("RIGHTSIZE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rightsize.ml_engine")

MODEL_TTL_SECONDS = int(os.getenv("RIGHTSIZE_MODEL_TTL_SECONDS", "600"))
PREDICTION_HORIZON_MINUTES = int(os.getenv("RIGHTSIZE_PREDICTION_HORIZON_MINUTES", "60"))
DEFAULT_CPU_HOUR_COST = float(os.getenv("RIGHTSIZE_CPU_HOUR_COST_INR", "3.5"))
SIMULATION_SPIKE_SECONDS = int(os.getenv("RIGHTSIZE_SIMULATION_SPIKE_SECONDS", "180"))
DEFAULT_SIMULATION_MODE = os.getenv("RIGHTSIZE_SIMULATION_MODE", "false").lower() in {
    "1",
    "true",
    "yes",
}
FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(title="RightSize AI Brain", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_prediction_history: deque[dict[str, Any]] = deque(maxlen=500)
_history_lock = threading.Lock()
_simulation_spikes: dict[str, datetime] = {}
_simulation_lock = threading.Lock()


def _cache_key(target_deployment: str, namespace: str | None) -> str:
    return f"{namespace or 'default'}:{target_deployment}"


def _simulation_spike_active(target_deployment: str, namespace: str | None) -> bool:
    key = _cache_key(target_deployment, namespace)
    with _simulation_lock:
        expires_at = _simulation_spikes.get(key)
        if not expires_at:
            return False
        if datetime.now(timezone.utc) >= expires_at:
            _simulation_spikes.pop(key, None)
            return False
        return True


def activate_simulation_spike(target_deployment: str, namespace: str | None) -> datetime:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SIMULATION_SPIKE_SECONDS)
    with _simulation_lock:
        _simulation_spikes[_cache_key(target_deployment, namespace)] = expires_at
    return expires_at


def generate_dummy_data(hours: int = 48, spike: bool = False) -> pd.DataFrame:
    """Generate seasonal demo traffic for local runs when Prometheus has no data."""
    now = datetime.now()
    dates = [now - timedelta(hours=i) for i in range(hours, 0, -1)]
    values = [
        500 + 300 * np.sin(i * np.pi / 12) + np.random.normal(0, 50)
        for i in range(hours)
    ]
    if spike:
        for index in range(max(0, hours - 8), hours):
            values[index] = values[index] * 1.7 + 260

    df = pd.DataFrame({"ds": dates, "y": values})
    df["y"] = df["y"].clip(lower=50)
    return df


def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    clean = df[["ds", "y"]].copy()
    clean["ds"] = pd.to_datetime(clean["ds"], errors="coerce")
    clean["y"] = pd.to_numeric(clean["y"], errors="coerce")
    clean = clean.dropna(subset=["ds", "y"])
    clean = clean.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")
    clean["y"] = clean["y"].clip(lower=0)
    return clean


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def calculate_confidence(yhat_lower: float, yhat_upper: float) -> float:
    if yhat_upper <= 0:
        return 0.0

    uncertainty_ratio = (yhat_upper - yhat_lower) / yhat_upper
    return round(_clamp(1 - uncertainty_ratio), 4)


def explain_prediction(
    *,
    confidence: float,
    latest_actual: float,
    predicted_cpu: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[str, str]:
    if latest_actual <= 0:
        trend_delta = 0.0
    else:
        trend_delta = (predicted_cpu - latest_actual) / latest_actual

    uncertainty_width = max(upper_bound - lower_bound, 0)

    if trend_delta >= 0.20:
        trend = "rising"
        trend_text = "CPU expected to increase due to a rising historical trend"
    elif trend_delta <= -0.20:
        trend = "falling"
        trend_text = "CPU expected to decrease after the recent demand peak"
    else:
        trend = "stable"
        trend_text = "CPU expected to stay near the current operating range"

    if confidence >= 0.80:
        confidence_text = "high confidence"
    elif confidence >= 0.55:
        confidence_text = "moderate confidence"
    else:
        confidence_text = "low confidence because recent variance is wide"

    if uncertainty_width > max(predicted_cpu, 1) * 0.75:
        return trend, f"{trend_text}; {confidence_text} with noticeable uncertainty."

    return trend, f"{trend_text}; {confidence_text} due to a consistent recent pattern."


def _frame_to_points(df: pd.DataFrame, limit: int = 288) -> list[dict[str, Any]]:
    if df.empty:
        return []

    frame = df.tail(limit).copy()
    frame["ds"] = pd.to_datetime(frame["ds"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame["y"] = frame["y"].round(2)
    return frame.rename(columns={"ds": "timestamp", "y": "cpu_millicores"}).to_dict("records")


@dataclass
class ModelEntry:
    model: Prophet
    trained_at: datetime
    source: str
    rows: int
    training_data: pd.DataFrame


class ProphetModelManager:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self._models: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()

    def _is_fresh(self, entry: ModelEntry) -> bool:
        return datetime.now(timezone.utc) - entry.trained_at < self.ttl

    def _load_training_data(
        self,
        target_deployment: str,
        namespace: str | None,
        simulate: bool,
    ) -> tuple[pd.DataFrame, str]:
        if simulate:
            logger.info("Simulation mode enabled for %s; skipping Prometheus fetch.", target_deployment)
            spike = _simulation_spike_active(target_deployment, namespace)
            return _clean_training_frame(generate_dummy_data(spike=spike)), "simulated-spike" if spike else "simulated"

        df = fetch_historical_cpu(target_deployment, days_back=2, namespace=namespace)
        df = _clean_training_frame(df)

        if len(df) < 2:
            logger.info("Prometheus data empty or too small for %s; using simulation data.", target_deployment)
            return _clean_training_frame(generate_dummy_data()), "simulated"

        return df, "prometheus"

    def _train(self, target_deployment: str, namespace: str | None, simulate: bool) -> ModelEntry:
        training_data, source = self._load_training_data(target_deployment, namespace, simulate)
        logger.info(
            "Retraining Prophet model for deployment=%s source=%s rows=%s",
            target_deployment,
            source,
            len(training_data),
        )

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )
        model.fit(training_data)
        return ModelEntry(
            model=model,
            trained_at=datetime.now(timezone.utc),
            source=source,
            rows=len(training_data),
            training_data=training_data,
        )

    def get_or_train(
        self,
        target_deployment: str,
        namespace: str | None = None,
        simulate: bool = False,
    ) -> tuple[ModelEntry, bool]:
        cache_mode = "simulated" if simulate else "prometheus"
        cache_key = f"{cache_mode}:{namespace or 'default'}:{target_deployment}"

        with self._lock:
            existing = self._models.get(cache_key)
            if existing and self._is_fresh(existing):
                logger.info("Reusing cached Prophet model for %s.", cache_key)
                return existing, False

            entry = self._train(target_deployment, namespace, simulate)
            self._models[cache_key] = entry
            return entry, True

    def invalidate(
        self,
        target_deployment: str,
        namespace: str | None = None,
        simulate: bool | None = None,
    ) -> None:
        namespace_key = namespace or "default"
        with self._lock:
            keys = list(self._models)
            for key in keys:
                mode, cached_namespace, cached_deployment = key.split(":", 2)
                if cached_deployment != target_deployment or cached_namespace != namespace_key:
                    continue
                if simulate is not None and mode != ("simulated" if simulate else "prometheus"):
                    continue
                self._models.pop(key, None)

    def predict(
        self,
        target_deployment: str,
        namespace: str | None = None,
        simulate: bool = DEFAULT_SIMULATION_MODE,
    ) -> dict[str, Any]:
        with self._lock:
            entry, retrained = self.get_or_train(target_deployment, namespace, simulate=simulate)
            periods = max(1, PREDICTION_HORIZON_MINUTES // 5)
            future = entry.model.make_future_dataframe(periods=periods, freq="5min")
            forecast = entry.model.predict(future)

        forecast_window = forecast.tail(periods).copy()
        peak_row = forecast_window.loc[forecast_window["yhat_upper"].idxmax()]

        predicted_cpu = max(float(peak_row["yhat"]), 0.0)
        lower_bound = max(float(peak_row["yhat_lower"]), 0.0)
        upper_bound = max(float(peak_row["yhat_upper"]), 0.0)
        recommended_cpu = int(max(round(upper_bound), 50))
        confidence = calculate_confidence(lower_bound, upper_bound)
        latest_actual = float(entry.training_data["y"].iloc[-1])
        trend, explanation = explain_prediction(
            confidence=confidence,
            latest_actual=latest_actual,
            predicted_cpu=predicted_cpu,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        result = {
            "target": target_deployment,
            "namespace": namespace,
            "recommended_cpu_limit": f"{recommended_cpu}m",
            "recommended_cpu_millicores": recommended_cpu,
            "current_cpu_millicores": round(latest_actual, 2),
            "confidence": confidence,
            "trend": trend,
            "explanation": explanation,
            "prediction": {
                "timestamp": pd.Timestamp(peak_row["ds"]).isoformat(),
                "yhat": round(predicted_cpu, 2),
                "yhat_lower": round(lower_bound, 2),
                "yhat_upper": round(upper_bound, 2),
            },
            "model": {
                "source": entry.source,
                "trained_at": entry.trained_at.isoformat(),
                "training_rows": entry.rows,
                "cache_ttl_seconds": int(self.ttl.total_seconds()),
                "retrained": retrained,
                "simulation_mode": simulate,
            },
            "status": "success",
        }

        with _history_lock:
            _prediction_history.append(
                {
                    **result,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        log_prediction(
            deployment=target_deployment,
            namespace=namespace,
            recommended_cpu_millicores=recommended_cpu,
            current_cpu_millicores=round(latest_actual, 2),
            confidence=confidence,
            trend=trend,
            explanation=explanation,
            source=entry.source,
            retrained=retrained,
        )

        return result

    def recent_metrics(
        self,
        target_deployment: str,
        namespace: str | None = None,
        limit: int = 288,
        simulate: bool = DEFAULT_SIMULATION_MODE,
    ) -> dict[str, Any]:
        entry, _ = self.get_or_train(target_deployment, namespace, simulate=simulate)
        return {
            "target": target_deployment,
            "namespace": namespace,
            "source": entry.source,
            "simulation_mode": simulate,
            "points": _frame_to_points(entry.training_data, limit=limit),
        }


model_manager = ProphetModelManager(ttl_seconds=MODEL_TTL_SECONDS)


def cpu_to_millicores(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    try:
        if text.endswith("m"):
            return int(float(text[:-1]))
        return int(float(text) * 1000)
    except ValueError:
        return default


def millicores_to_cpu(value: int) -> str:
    return f"{max(int(value), 1)}m"


def _load_k8s_api() -> Any:
    if k8s is None or k8s_config is None:
        raise RuntimeError("Kubernetes client is not installed.")

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    return k8s.AppsV1Api()


def _read_deployment_cpu(api: Any, deployment: str, namespace: str) -> tuple[dict[str, Any], str, str]:
    obj = api.read_namespaced_deployment(name=deployment, namespace=namespace)
    container = obj.spec.template.spec.containers[0]
    resources = container.resources
    limits = resources.limits or {}
    requests_map = resources.requests or {}
    current = limits.get("cpu") or requests_map.get("cpu") or "unset"
    return obj, container.name, str(current)


def _patch_deployment_annotations(
    *,
    deployment: str,
    namespace: str,
    annotations: dict[str, str | None],
) -> None:
    api = _load_k8s_api()
    body = {"metadata": {"annotations": annotations}}
    api.patch_namespaced_deployment(name=deployment, namespace=namespace, body=body)


def _patch_deployment_cpu(
    *,
    deployment: str,
    namespace: str,
    new_cpu: str,
) -> str:
    api = _load_k8s_api()
    obj, container_name, old_cpu = _read_deployment_cpu(api, deployment, namespace)
    container = obj.spec.template.spec.containers[0]
    resources = container.resources.to_dict() if container.resources else {}
    limits = dict(resources.get("limits") or {})
    limits["cpu"] = new_cpu
    resources["limits"] = limits

    body = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container_name,
                            "resources": resources,
                        }
                    ]
                }
            }
        }
    }
    api.patch_namespaced_deployment(name=deployment, namespace=namespace, body=body)
    return old_cpu


def _k8s_error_response(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "Kubernetes is not reachable. Start Docker Desktop, Minikube, and make sure "
            f"kubectl uses the minikube context. Details: {exc}"
        ),
    )


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "message": "Welcome to the RightSize AI Brain!",
        "status": "online",
        "endpoints": [
            "/predict?target_deployment=demo-app",
            "/predict?target_deployment=demo-app&simulate=true",
            "/dashboard",
            "/metrics?target_deployment=demo-app",
            "/prediction-history",
            "/scaling-history",
            "/operator-decisions",
            "/simulate/spike",
            "/autoscaling/enable",
            "/apply-recommendation",
            "/cost-estimate?target_deployment=demo-app&allocated_cpu_millicores=1000",
        ],
    }


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/predict")
def predict_resource_needs(
    target_deployment: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    simulate: bool = Query(default=DEFAULT_SIMULATION_MODE),
) -> dict[str, Any]:
    if target_deployment is None:
        return {
            "error": "missing_target",
            "message": "Tell the AI which deployment to inspect.",
            "hint": "Try /predict?target_deployment=demo-app",
            "status": "error",
        }

    logger.info("Received prediction request for deployment=%s namespace=%s", target_deployment, namespace)
    return model_manager.predict(
        target_deployment=target_deployment,
        namespace=namespace,
        simulate=simulate,
    )


@app.get("/metrics")
def metrics(
    target_deployment: str = Query(default="demo-app"),
    namespace: str | None = Query(default=None),
    limit: int = Query(default=288, ge=1, le=2000),
    simulate: bool = Query(default=DEFAULT_SIMULATION_MODE),
) -> dict[str, Any]:
    return model_manager.recent_metrics(
        target_deployment,
        namespace=namespace,
        limit=limit,
        simulate=simulate,
    )


@app.get("/prediction-history")
def prediction_history(
    target_deployment: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return {
        "items": read_prediction_history(
            limit=limit,
            deployment=target_deployment,
            namespace=namespace,
        )
    }


@app.get("/scaling-history")
def scaling_history(
    deployment: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    return {
        "items": read_scaling_history(
            limit=limit,
            deployment=deployment,
            namespace=namespace,
        )
    }


@app.get("/operator-decisions")
def operator_decisions(
    deployment: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    return {
        "items": read_operator_decisions(
            limit=limit,
            deployment=deployment,
            namespace=namespace,
        )
    }


@app.get("/deployments")
def deployments(namespace: str = Query(default="default")) -> dict[str, Any]:
    try:
        api = _load_k8s_api()
        items = api.list_namespaced_deployment(namespace=namespace).items
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": str(exc),
            "items": [{"name": "demo-app", "namespace": namespace, "source": "demo-fallback"}],
        }

    return {
        "status": "success",
        "items": [
            {
                "name": item.metadata.name,
                "namespace": namespace,
                "replicas": item.spec.replicas,
                "available_replicas": item.status.available_replicas or 0,
            }
            for item in items
        ],
    }


@app.post("/simulate/spike")
def simulate_spike(
    target_deployment: str = Query(default="demo-app"),
    namespace: str | None = Query(default=None),
) -> dict[str, Any]:
    expires_at = activate_simulation_spike(target_deployment, namespace)
    model_manager.invalidate(target_deployment, namespace=namespace, simulate=True)
    log_operator_decision(
        deployment=target_deployment,
        namespace=namespace or "default",
        action="simulation_spike",
        reason="Demo CPU spike activated for the synthetic workload.",
        metadata={"expires_at": expires_at.isoformat()},
    )
    return {
        "status": "success",
        "target": target_deployment,
        "namespace": namespace,
        "expires_at": expires_at.isoformat(),
        "message": "Synthetic CPU spike activated. The next refresh will retrain simulated data.",
    }


@app.post("/autoscaling/enable")
def enable_autoscaling(
    target_deployment: str = Query(default="demo-app"),
    namespace: str = Query(default="default"),
    mode: str = Query(default="auto", pattern="^(auto|recommendation)$"),
    min_cpu: str = Query(default="100m"),
    max_cpu: str = Query(default="2000m"),
    cooldown_seconds: int = Query(default=60, ge=1),
    change_threshold_percent: float = Query(default=5, ge=0),
    dry_run: bool = Query(default=False),
) -> dict[str, Any]:
    annotations = {
        "rightsize.ai/enabled": "true",
        "rightsize.ai/mode": mode,
        "rightsize.ai/min-cpu": min_cpu,
        "rightsize.ai/max-cpu": max_cpu,
        "rightsize.ai/cooldown-seconds": str(cooldown_seconds),
        "rightsize.ai/change-threshold-percent": str(change_threshold_percent),
    }

    if not dry_run:
        try:
            _patch_deployment_annotations(
                deployment=target_deployment,
                namespace=namespace,
                annotations=annotations,
            )
        except Exception as exc:
            raise _k8s_error_response(exc) from exc

    config_event = upsert_deployment_config(
        deployment=target_deployment,
        namespace=namespace,
        enabled=True,
        mode=mode,
        min_cpu=min_cpu,
        max_cpu=max_cpu,
        cooldown_seconds=cooldown_seconds,
        change_threshold_percent=change_threshold_percent,
    )
    log_operator_decision(
        deployment=target_deployment,
        namespace=namespace,
        action="autoscaling_enabled" if not dry_run else "autoscaling_enabled_demo",
        reason=f"Autoscaling enabled in {mode} mode.",
        metadata={"dry_run": dry_run, "annotations": annotations},
    )
    return {"status": "success", "dry_run": dry_run, "config": config_event}


@app.post("/autoscaling/disable")
def disable_autoscaling(
    target_deployment: str = Query(default="demo-app"),
    namespace: str = Query(default="default"),
    dry_run: bool = Query(default=False),
) -> dict[str, Any]:
    annotations = {"rightsize.ai/enabled": "false"}
    if not dry_run:
        try:
            _patch_deployment_annotations(
                deployment=target_deployment,
                namespace=namespace,
                annotations=annotations,
            )
        except Exception as exc:
            raise _k8s_error_response(exc) from exc

    config_event = upsert_deployment_config(
        deployment=target_deployment,
        namespace=namespace,
        enabled=False,
    )
    log_operator_decision(
        deployment=target_deployment,
        namespace=namespace,
        action="autoscaling_disabled" if not dry_run else "autoscaling_disabled_demo",
        reason="Autoscaling disabled for this deployment.",
        metadata={"dry_run": dry_run},
    )
    return {"status": "success", "dry_run": dry_run, "config": config_event}


@app.post("/apply-recommendation")
def apply_recommendation(
    target_deployment: str = Query(default="demo-app"),
    namespace: str = Query(default="default"),
    allocated_cpu_millicores: int = Query(default=1000, ge=1),
    simulate: bool = Query(default=DEFAULT_SIMULATION_MODE),
    dry_run: bool = Query(default=False),
) -> dict[str, Any]:
    prediction = model_manager.predict(
        target_deployment=target_deployment,
        namespace=namespace,
        simulate=simulate,
    )
    new_cpu = prediction["recommended_cpu_limit"]
    old_cpu = millicores_to_cpu(allocated_cpu_millicores)

    if not dry_run:
        try:
            old_cpu = _patch_deployment_cpu(
                deployment=target_deployment,
                namespace=namespace,
                new_cpu=new_cpu,
            )
        except Exception as exc:
            raise _k8s_error_response(exc) from exc

    source = "dashboard-demo" if dry_run else "dashboard"
    reason = f"Manual apply from dashboard: {prediction['explanation']}"
    log_scaling_action(
        deployment=target_deployment,
        namespace=namespace,
        old_cpu=old_cpu,
        new_cpu=new_cpu,
        confidence=prediction["confidence"],
        reason=reason,
        source=source,
    )
    log_operator_decision(
        deployment=target_deployment,
        namespace=namespace,
        action="manual_apply_demo" if dry_run else "manual_apply",
        reason=reason,
        confidence=prediction["confidence"],
        current_cpu=old_cpu,
        recommended_cpu=prediction["recommended_cpu_limit"],
        desired_cpu=new_cpu,
        metadata={"dry_run": dry_run, "simulate": simulate},
    )

    return {
        "status": "success",
        "dry_run": dry_run,
        "old_cpu": old_cpu,
        "new_cpu": new_cpu,
        "prediction": prediction,
    }


@app.get("/cost-estimate")
def cost_estimate(
    target_deployment: str = Query(default="demo-app"),
    namespace: str | None = Query(default=None),
    allocated_cpu_millicores: int = Query(default=1000, ge=1),
    cpu_hour_cost: float = Query(default=DEFAULT_CPU_HOUR_COST, ge=0),
    simulate: bool = Query(default=DEFAULT_SIMULATION_MODE),
) -> dict[str, Any]:
    prediction = model_manager.predict(
        target_deployment=target_deployment,
        namespace=namespace,
        simulate=simulate,
    )
    recommended = int(prediction["recommended_cpu_millicores"])
    saved_millicores = max(allocated_cpu_millicores - recommended, 0)
    saved_cpu_hours_per_month = (saved_millicores / 1000) * 24 * 30
    monthly_savings = saved_cpu_hours_per_month * cpu_hour_cost

    return {
        "target": target_deployment,
        "namespace": namespace,
        "allocated_cpu_millicores": allocated_cpu_millicores,
        "recommended_cpu_millicores": recommended,
        "saved_cpu_millicores": saved_millicores,
        "estimated_saved_cpu_hours_per_month": round(saved_cpu_hours_per_month, 2),
        "estimated_monthly_savings": round(monthly_savings, 2),
        "currency": "INR",
        "cpu_hour_cost": cpu_hour_cost,
        "simulation_mode": simulate,
        "explanation": prediction["explanation"],
    }


@app.get("/simulation-mode")
def simulation_mode() -> dict[str, Any]:
    return {
        "enabled_by_default": DEFAULT_SIMULATION_MODE,
        "toggle_hint": "Append ?simulate=true to /predict, /metrics, or /cost-estimate for demo data.",
    }
