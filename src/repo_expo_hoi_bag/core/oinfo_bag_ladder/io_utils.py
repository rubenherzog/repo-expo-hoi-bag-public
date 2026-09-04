from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def _install_pyarrow_unregister_hotfix() -> None:
    try:
        import pyarrow  # type: ignore
    except Exception:
        return

    if getattr(pyarrow, "_oinfo_unregister_hotfix", False):
        return

    original = pyarrow.unregister_extension_type

    def _safe_unregister(name: str):
        try:
            return original(name)
        except Exception as e:
            msg = str(e)
            if name == "arrow.py_extension_type" and "No type extension with name" in msg:
                return None
            raise

    pyarrow.unregister_extension_type = _safe_unregister
    pyarrow._oinfo_unregister_hotfix = True


def _probe_parquet_engine(engine: str) -> Tuple[bool, str | None]:
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="oinfo_parquet_probe_"))
    try:
        probe_path = tmpdir / "probe.parquet"
        pd.DataFrame({"_probe": [1.0]}).to_parquet(probe_path, index=False, engine=engine)
        _ = pd.read_parquet(probe_path, engine=engine)
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def ensure_parquet_support(preferred: Tuple[str, ...] = ("pyarrow", "fastparquet")) -> Tuple[str, Dict[str, str]]:
    errors: Dict[str, str] = {}
    for engine in preferred:
        try:
            if engine == "pyarrow":
                import pyarrow  # noqa: F401

                _install_pyarrow_unregister_hotfix()
            elif engine == "fastparquet":
                import fastparquet  # noqa: F401
            else:
                errors[engine] = "unsupported engine"
                continue
        except Exception as e:
            errors[engine] = f"import failed: {e}"
            continue

        ok, err = _probe_parquet_engine(engine)
        if ok:
            return engine, errors
        errors[engine] = f"probe failed: {err}"

    details = "; ".join([f"{k} -> {v}" for k, v in errors.items()])
    raise RuntimeError("No working parquet engine found. " + details)


PARQUET_ENGINE = None
PARQUET_ENGINE_ERRORS: Dict[str, str] = {}


def _engine_order(primary: str) -> List[str]:
    order = [primary]
    for e in ("fastparquet", "pyarrow"):
        if e != primary:
            order.append(e)
    return order


def write_parquet(df: pd.DataFrame | None, path: Path) -> None:
    global PARQUET_ENGINE, PARQUET_ENGINE_ERRORS

    if PARQUET_ENGINE is None:
        PARQUET_ENGINE, PARQUET_ENGINE_ERRORS = ensure_parquet_support()

    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()

    errors = {}
    for engine in _engine_order(PARQUET_ENGINE):
        try:
            df.to_parquet(path, index=False, engine=engine)
            if engine != PARQUET_ENGINE:
                PARQUET_ENGINE = engine
            return
        except Exception as e:
            errors[engine] = str(e)

    raise RuntimeError(f"Failed to write parquet to {path}. Errors: {errors}")


def read_parquet(path: Path) -> pd.DataFrame:
    global PARQUET_ENGINE, PARQUET_ENGINE_ERRORS

    if PARQUET_ENGINE is None:
        PARQUET_ENGINE, PARQUET_ENGINE_ERRORS = ensure_parquet_support()

    errors = {}
    for engine in _engine_order(PARQUET_ENGINE):
        try:
            out = pd.read_parquet(path, engine=engine)
            if engine != PARQUET_ENGINE:
                PARQUET_ENGINE = engine
            return out
        except Exception as e:
            errors[engine] = str(e)

    raise RuntimeError(f"Failed to read parquet from {path}. Errors: {errors}")
