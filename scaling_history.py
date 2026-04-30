import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_FILE = Path(os.getenv("RIGHTSIZE_DB_FILE", "rightsize.db"))
LEGACY_HISTORY_FILE = Path(os.getenv("RIGHTSIZE_HISTORY_FILE", "scaling_history.json"))
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS scaling_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            deployment TEXT NOT NULL,
            namespace TEXT NOT NULL,
            old_cpu TEXT NOT NULL,
            new_cpu TEXT NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'operator'
        );

        CREATE TABLE IF NOT EXISTS operator_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            deployment TEXT NOT NULL,
            namespace TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            confidence REAL,
            current_cpu TEXT,
            recommended_cpu TEXT,
            desired_cpu TEXT,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            deployment TEXT NOT NULL,
            namespace TEXT,
            recommended_cpu_millicores INTEGER NOT NULL,
            current_cpu_millicores REAL NOT NULL,
            confidence REAL NOT NULL,
            trend TEXT NOT NULL,
            explanation TEXT NOT NULL,
            source TEXT NOT NULL,
            retrained INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS deployment_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            deployment TEXT NOT NULL,
            namespace TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'recommendation',
            min_cpu TEXT,
            max_cpu TEXT,
            cooldown_seconds INTEGER,
            change_threshold_percent REAL,
            UNIQUE(deployment, namespace)
        );
        """
    )
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    if "metadata" in item:
        try:
            item["metadata"] = json.loads(item["metadata"] or "{}")
        except json.JSONDecodeError:
            item["metadata"] = {}
    if "retrained" in item:
        item["retrained"] = bool(item["retrained"])
    if "enabled" in item:
        item["enabled"] = bool(item["enabled"])
    return item


def _migrate_legacy_json(conn: sqlite3.Connection) -> None:
    if not LEGACY_HISTORY_FILE.exists():
        return

    existing = conn.execute("SELECT COUNT(*) FROM scaling_actions").fetchone()[0]
    if existing:
        return

    try:
        with LEGACY_HISTORY_FILE.open("r", encoding="utf-8") as handle:
            events = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(events, list):
        return

    for event in events:
        if not isinstance(event, dict):
            continue
        conn.execute(
            """
            INSERT INTO scaling_actions
            (timestamp, deployment, namespace, old_cpu, new_cpu, confidence, reason, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("timestamp") or _now(),
                event.get("deployment") or "unknown",
                event.get("namespace") or "default",
                event.get("old_cpu") or "unset",
                event.get("new_cpu") or "unset",
                float(event.get("confidence") or 0),
                event.get("reason") or "Migrated legacy scaling action",
                "legacy-json",
            ),
        )
    conn.commit()


def init_history_store() -> None:
    with _LOCK, _connect() as conn:
        _init_db(conn)
        _migrate_legacy_json(conn)


def read_scaling_history(
    limit: int = 100,
    deployment: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM scaling_actions WHERE 1=1"
    params: list[Any] = []
    if deployment:
        query += " AND deployment = ?"
        params.append(deployment)
    if namespace:
        query += " AND namespace = ?"
        params.append(namespace)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _LOCK, _connect() as conn:
        _init_db(conn)
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(row) for row in reversed(rows)]


def log_scaling_action(
    *,
    deployment: str,
    namespace: str,
    old_cpu: str,
    new_cpu: str,
    confidence: float,
    reason: str,
    source: str = "operator",
) -> dict[str, Any]:
    event = {
        "timestamp": _now(),
        "deployment": deployment,
        "namespace": namespace,
        "old_cpu": old_cpu,
        "new_cpu": new_cpu,
        "confidence": round(float(confidence), 4),
        "reason": reason,
        "source": source,
    }

    with _LOCK, _connect() as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT INTO scaling_actions
            (timestamp, deployment, namespace, old_cpu, new_cpu, confidence, reason, source)
            VALUES (:timestamp, :deployment, :namespace, :old_cpu, :new_cpu, :confidence, :reason, :source)
            """,
            event,
        )
        conn.commit()

    return event


def log_operator_decision(
    *,
    deployment: str,
    namespace: str,
    action: str,
    reason: str,
    confidence: float | None = None,
    current_cpu: str | None = None,
    recommended_cpu: str | None = None,
    desired_cpu: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "timestamp": _now(),
        "deployment": deployment,
        "namespace": namespace,
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "current_cpu": current_cpu,
        "recommended_cpu": recommended_cpu,
        "desired_cpu": desired_cpu,
        "metadata": json.dumps(metadata or {}, sort_keys=True),
    }

    with _LOCK, _connect() as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT INTO operator_decisions
            (timestamp, deployment, namespace, action, reason, confidence, current_cpu,
             recommended_cpu, desired_cpu, metadata)
            VALUES (:timestamp, :deployment, :namespace, :action, :reason, :confidence,
                    :current_cpu, :recommended_cpu, :desired_cpu, :metadata)
            """,
            event,
        )
        conn.commit()

    event["metadata"] = metadata or {}
    return event


def read_operator_decisions(
    limit: int = 100,
    deployment: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM operator_decisions WHERE 1=1"
    params: list[Any] = []
    if deployment:
        query += " AND deployment = ?"
        params.append(deployment)
    if namespace:
        query += " AND namespace = ?"
        params.append(namespace)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _LOCK, _connect() as conn:
        _init_db(conn)
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(row) for row in reversed(rows)]


def log_prediction(
    *,
    deployment: str,
    namespace: str | None,
    recommended_cpu_millicores: int,
    current_cpu_millicores: float,
    confidence: float,
    trend: str,
    explanation: str,
    source: str,
    retrained: bool,
) -> dict[str, Any]:
    event = {
        "timestamp": _now(),
        "deployment": deployment,
        "namespace": namespace,
        "recommended_cpu_millicores": int(recommended_cpu_millicores),
        "current_cpu_millicores": float(current_cpu_millicores),
        "confidence": round(float(confidence), 4),
        "trend": trend,
        "explanation": explanation,
        "source": source,
        "retrained": 1 if retrained else 0,
    }

    with _LOCK, _connect() as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT INTO predictions
            (timestamp, deployment, namespace, recommended_cpu_millicores,
             current_cpu_millicores, confidence, trend, explanation, source, retrained)
            VALUES (:timestamp, :deployment, :namespace, :recommended_cpu_millicores,
                    :current_cpu_millicores, :confidence, :trend, :explanation, :source, :retrained)
            """,
            event,
        )
        conn.commit()

    event["retrained"] = bool(event["retrained"])
    return event


def read_prediction_history(
    limit: int = 100,
    deployment: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM predictions WHERE 1=1"
    params: list[Any] = []
    if deployment:
        query += " AND deployment = ?"
        params.append(deployment)
    if namespace:
        query += " AND namespace = ?"
        params.append(namespace)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _LOCK, _connect() as conn:
        _init_db(conn)
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(row) for row in reversed(rows)]


def upsert_deployment_config(
    *,
    deployment: str,
    namespace: str,
    enabled: bool,
    mode: str = "recommendation",
    min_cpu: str | None = None,
    max_cpu: str | None = None,
    cooldown_seconds: int | None = None,
    change_threshold_percent: float | None = None,
) -> dict[str, Any]:
    event = {
        "timestamp": _now(),
        "deployment": deployment,
        "namespace": namespace,
        "enabled": 1 if enabled else 0,
        "mode": mode,
        "min_cpu": min_cpu,
        "max_cpu": max_cpu,
        "cooldown_seconds": cooldown_seconds,
        "change_threshold_percent": change_threshold_percent,
    }

    with _LOCK, _connect() as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT INTO deployment_configs
            (timestamp, deployment, namespace, enabled, mode, min_cpu, max_cpu,
             cooldown_seconds, change_threshold_percent)
            VALUES (:timestamp, :deployment, :namespace, :enabled, :mode, :min_cpu,
                    :max_cpu, :cooldown_seconds, :change_threshold_percent)
            ON CONFLICT(deployment, namespace) DO UPDATE SET
                timestamp = excluded.timestamp,
                enabled = excluded.enabled,
                mode = excluded.mode,
                min_cpu = COALESCE(excluded.min_cpu, deployment_configs.min_cpu),
                max_cpu = COALESCE(excluded.max_cpu, deployment_configs.max_cpu),
                cooldown_seconds = COALESCE(excluded.cooldown_seconds, deployment_configs.cooldown_seconds),
                change_threshold_percent = COALESCE(excluded.change_threshold_percent, deployment_configs.change_threshold_percent)
            """,
            event,
        )
        conn.commit()

    event["enabled"] = bool(event["enabled"])
    return event


init_history_store()
