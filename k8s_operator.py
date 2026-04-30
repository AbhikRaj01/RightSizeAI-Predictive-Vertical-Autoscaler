import logging
import os
from datetime import datetime, timezone
from typing import Any

import kopf
import kubernetes.client as k8s
import requests
from kubernetes import config

from scaling_history import log_operator_decision, log_scaling_action


logger = logging.getLogger("rightsize.operator")

AI_BRAIN_URL = os.getenv("RIGHTSIZE_AI_BRAIN_URL", "http://127.0.0.1:8000")
DEFAULT_COOLDOWN_SECONDS = int(os.getenv("RIGHTSIZE_COOLDOWN_SECONDS", "300"))
DEFAULT_CHANGE_THRESHOLD_PERCENT = float(os.getenv("RIGHTSIZE_CHANGE_THRESHOLD_PERCENT", "15"))
DEFAULT_MIN_CPU = os.getenv("RIGHTSIZE_MIN_CPU", "100m")
DEFAULT_MAX_CPU = os.getenv("RIGHTSIZE_MAX_CPU", "4000m")
MIN_CONFIDENCE = float(os.getenv("RIGHTSIZE_MIN_CONFIDENCE", "0.80"))

ANNOTATION_LAST_SCALED_AT = "rightsize.ai/last-scaled-at"
_last_scaled_at: dict[str, datetime] = {}


def _load_kubernetes_config() -> None:
    try:
        config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes config.")
        return
    except Exception:
        pass

    try:
        config.load_kube_config()
        logger.info("Loaded local Kubernetes config.")
    except Exception as exc:
        logger.warning("Kubernetes config could not be loaded yet: %s", exc)


_load_kubernetes_config()


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


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _annotation_int(annotations: dict[str, str], key: str, default_value: int) -> int:
    value = annotations.get(key)
    try:
        return int(value) if value is not None else default_value
    except ValueError:
        return default_value


def _annotation_float(annotations: dict[str, str], key: str, default_value: float) -> float:
    value = annotations.get(key)
    try:
        return float(value) if value is not None else default_value
    except ValueError:
        return default_value


def _cooldown_active(
    *,
    deployment_key: str,
    annotations: dict[str, str],
    cooldown_seconds: int,
    now: datetime,
) -> tuple[bool, int]:
    last_scaled = _parse_timestamp(annotations.get(ANNOTATION_LAST_SCALED_AT))
    last_scaled = last_scaled or _last_scaled_at.get(deployment_key)
    if last_scaled is None:
        return False, 0

    elapsed = (now - last_scaled).total_seconds()
    remaining = int(cooldown_seconds - elapsed)
    return remaining > 0, max(remaining, 0)


def _current_container(spec: dict[str, Any]) -> dict[str, Any]:
    containers = spec.get("template", {}).get("spec", {}).get("containers", [])
    if not containers:
        raise ValueError("Deployment has no containers to resize.")
    return containers[0]


def _current_cpu_millicores(container: dict[str, Any]) -> int | None:
    resources = container.get("resources", {})
    limits = resources.get("limits", {})
    requests_map = resources.get("requests", {})
    return cpu_to_millicores(limits.get("cpu")) or cpu_to_millicores(requests_map.get("cpu"))


def _bounded_cpu(
    *,
    recommended_millicores: int,
    annotations: dict[str, str],
) -> tuple[int, int, int]:
    min_cpu = cpu_to_millicores(annotations.get("rightsize.ai/min-cpu"), cpu_to_millicores(DEFAULT_MIN_CPU, 100))
    max_cpu = cpu_to_millicores(annotations.get("rightsize.ai/max-cpu"), cpu_to_millicores(DEFAULT_MAX_CPU, 4000))

    if min_cpu is None:
        min_cpu = 100
    if max_cpu is None:
        max_cpu = 4000
    if max_cpu < min_cpu:
        max_cpu = min_cpu

    return max(min(recommended_millicores, max_cpu), min_cpu), min_cpu, max_cpu


def _should_scale(
    *,
    current_millicores: int | None,
    desired_millicores: int,
    threshold_percent: float,
) -> tuple[bool, float]:
    if current_millicores is None or current_millicores <= 0:
        return True, 100.0

    change_percent = abs(desired_millicores - current_millicores) / current_millicores * 100
    return change_percent >= threshold_percent, change_percent


def _record_decision(logger, **kwargs) -> None:
    try:
        log_operator_decision(**kwargs)
    except Exception as exc:
        logger.warning("Could not record operator decision: %s", exc)


@kopf.on.timer(
    "apps",
    "v1",
    "deployments",
    interval=20,
    annotations={"rightsize.ai/enabled": "true"},
)
def rightsize_deployment(
    name,
    namespace,
    spec,
    annotations,
    logger,
    **kwargs,
):
    deployment_key = f"{namespace}/{name}"
    annotations = annotations or {}
    now = datetime.now(timezone.utc)

    cooldown_seconds = _annotation_int(
        annotations,
        "rightsize.ai/cooldown-seconds",
        DEFAULT_COOLDOWN_SECONDS,
    )
    threshold_percent = _annotation_float(
        annotations,
        "rightsize.ai/change-threshold-percent",
        DEFAULT_CHANGE_THRESHOLD_PERCENT,
    )

    logger.info("Analyzing %s for predictive rightsizing.", deployment_key)

    mode = annotations.get("rightsize.ai/mode", "auto")
    if mode == "recommendation":
        message = "Recommendation mode active; dashboard approval required before resizing."
        logger.info("Skipping %s: %s", deployment_key, message)
        _record_decision(
            logger,
            deployment=name,
            namespace=namespace,
            action="skipped_recommendation_mode",
            reason=message,
        )
        return

    active, remaining = _cooldown_active(
        deployment_key=deployment_key,
        annotations=annotations,
        cooldown_seconds=cooldown_seconds,
        now=now,
    )
    if active:
        message = f"Cooldown active for {remaining}s."
        logger.info("Skipping %s: %s", deployment_key, message)
        _record_decision(
            logger,
            deployment=name,
            namespace=namespace,
            action="skipped_cooldown",
            reason=message,
            metadata={"remaining_seconds": remaining},
        )
        return

    try:
        response = requests.get(
            f"{AI_BRAIN_URL}/predict",
            params={"target_deployment": name, "namespace": namespace},
            timeout=8,
        )
        response.raise_for_status()
        prediction = response.json()

        if prediction.get("status") != "success":
            message = f"Prediction failed: {prediction}"
            logger.info("Skipping %s: %s", deployment_key, message)
            _record_decision(
                logger,
                deployment=name,
                namespace=namespace,
                action="skipped_prediction_failed",
                reason=message,
                metadata={"prediction": prediction},
            )
            return

        confidence = float(prediction.get("confidence", 0))
        if confidence < MIN_CONFIDENCE:
            message = f"Confidence {confidence:.2f} below required {MIN_CONFIDENCE:.2f}."
            logger.info("Skipping %s: %s", deployment_key, message)
            _record_decision(
                logger,
                deployment=name,
                namespace=namespace,
                action="skipped_low_confidence",
                reason=message,
                confidence=confidence,
                recommended_cpu=prediction.get("recommended_cpu_limit"),
            )
            return

        recommended = cpu_to_millicores(
            prediction.get("recommended_cpu_millicores")
            or prediction.get("recommended_cpu_limit")
        )
        if recommended is None:
            message = "Prediction did not include a valid CPU recommendation."
            logger.info("Skipping %s: %s", deployment_key, message)
            _record_decision(
                logger,
                deployment=name,
                namespace=namespace,
                action="skipped_invalid_recommendation",
                reason=message,
                confidence=confidence,
                metadata={"prediction": prediction},
            )
            return

        desired, min_cpu, max_cpu = _bounded_cpu(
            recommended_millicores=recommended,
            annotations=annotations,
        )
        container = _current_container(spec)
        container_name = container["name"]
        current = _current_cpu_millicores(container)
        should_scale, change_percent = _should_scale(
            current_millicores=current,
            desired_millicores=desired,
            threshold_percent=threshold_percent,
        )

        if not should_scale:
            message = f"CPU change {change_percent:.1f}% is below {threshold_percent:.1f}% threshold."
            logger.info("Skipping %s: %s", deployment_key, message)
            _record_decision(
                logger,
                deployment=name,
                namespace=namespace,
                action="skipped_below_threshold",
                reason=message,
                confidence=confidence,
                current_cpu=millicores_to_cpu(current) if current is not None else "unset",
                recommended_cpu=prediction.get("recommended_cpu_limit"),
                desired_cpu=millicores_to_cpu(desired),
                metadata={"change_percent": change_percent, "threshold_percent": threshold_percent},
            )
            return

        resources = dict(container.get("resources", {}))
        limits = dict(resources.get("limits", {}))
        limits["cpu"] = millicores_to_cpu(desired)
        resources["limits"] = limits

        scaled_at = now.isoformat()
        patch = {
            "metadata": {
                "annotations": {
                    ANNOTATION_LAST_SCALED_AT: scaled_at,
                }
            },
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
            },
        }

        old_cpu = millicores_to_cpu(current) if current is not None else "unset"
        new_cpu = millicores_to_cpu(desired)
        reason = prediction.get("explanation", "Predictive CPU adjustment")

        api = k8s.AppsV1Api()
        api.patch_namespaced_deployment(name=name, namespace=namespace, body=patch)
        _last_scaled_at[deployment_key] = now
        log_scaling_action(
            deployment=name,
            namespace=namespace,
            old_cpu=old_cpu,
            new_cpu=new_cpu,
            confidence=confidence,
            reason=reason,
        )
        _record_decision(
            logger,
            deployment=name,
            namespace=namespace,
            action="scaled",
            reason=reason,
            confidence=confidence,
            current_cpu=old_cpu,
            recommended_cpu=prediction.get("recommended_cpu_limit"),
            desired_cpu=new_cpu,
            metadata={"change_percent": change_percent, "min_cpu": min_cpu, "max_cpu": max_cpu},
        )

        logger.info(
            "Scaled %s from %s to %s; confidence=%.2f, change=%.1f%%, bounds=%sm-%sm.",
            deployment_key,
            old_cpu,
            new_cpu,
            confidence,
            change_percent,
            min_cpu,
            max_cpu,
        )

    except Exception as exc:
        logger.error("Failed to process %s: %s", deployment_key, exc)
        _record_decision(
            logger,
            deployment=name,
            namespace=namespace,
            action="error",
            reason=str(exc),
        )
