import numpy as np
from typing import Optional, Tuple, Literal
from .constants import MAX_DATA
from .gating import TimeGate, gate_indices


def is_overflow(raw_u16 : np.ndarray, margin_codes : int = 2) -> bool:
    if raw_u16.size == 0:
        return False
    raw_min = int(np.min(raw_u16))
    raw_max = int(np.max(raw_u16))
    return raw_min <= margin_codes or raw_max >= (MAX_DATA - margin_codes)


def _first_run_start(mask : np.ndarray, min_run : int) -> Optional[int]:
    """
    Вернуть индекс начала первой серии True длиной >= min_run
    """
    if min_run < 1:
        raise ValueError("min_run must be >= 1")

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    if min_run == 1:
        return int(idx[0])

    run_start = int(idx[0])
    prev = int(idx[0])
    run_len = 1

    for k in idx[1:]:
        k = int(k)
        if k == prev + 1:
            run_len += 1
        else:
            run_start = k
            run_len = 1
        prev = k
        if run_len >= min_run:
            return int(run_start)
    return None


def estimate_edge_index(
    y : np.ndarray,
    *,
    frac : float = 0.5,
    t_s : Optional[np.ndarray] = None,
    baseline_head_fraction : float = 0.1,
    mode : Literal["rise", "fall", "abs"] = "rise",
    min_run : int = 2,
) -> Optional[int]:
    """
    Простая оценка индекса фронта:
    - baseline берётся по началу записи
    - порог = frac * характерный пик
    - mode задаёт полярность
    - min_run защищает от одиночных spike'ов
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return None
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if not (0.0 < baseline_head_fraction <= 1.0):
        raise ValueError(f"baseline_head_fraction must be in (0, 1], got {baseline_head_fraction}")
    if t_s is not None:
        t_s = np.asarray(t_s, dtype=np.float64)
        if t_s.shape != y.shape:
            raise ValueError("t_s and y must have the same shape")
    
    if t_s is not None and np.any(t_s < 0):
        baseline = float(np.mean(y[t_s < 0.0]))
    else:
        n_head = max(1, int(round(y.size * baseline_head_fraction)))
        baseline = float(np.mean(y[:n_head]))

    yc = y - baseline

    if mode == "rise":
        peak = float(np.max(yc))
        if peak <= 0.0:
            return None
        thr = frac * peak
        mask = yc >= thr
    elif mode == "fall":
        peak = float(np.min(yc))  # отрицательный
        if peak >= 0.0:
            return None
        thr = frac * peak
        mask = yc <= thr
    elif mode == "abs":
        peak_abs = float(np.max(np.abs(yc)))
        if peak_abs <= 0.0:
            return None
        thr = frac * peak_abs
        mask = np.abs(yc) >= thr
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    return _first_run_start(mask, min_run=min_run)


def estimate_ref_index_in_gate(
    t_s : np.ndarray,
    ref_codes : np.ndarray,
    gate : TimeGate,
    *,
    frac : float,
    pretrigger_percent : int,
    mode : Literal["rise", "fall", "abs"] = "rise",
    min_run : int = 2,
) -> Optional[int]:
    idx_gate = gate_indices(t_s, gate)
    if idx_gate.size == 0:
        return None

    ref_seg = ref_codes[idx_gate]
    t_seg = t_s[idx_gate]

    idx_local = estimate_edge_index(
        ref_seg,
        frac=frac,
        t_s=t_seg,
        baseline_head_fraction=max(0.01, pretrigger_percent / 100.0),
        mode=mode,
        min_run=min_run,
    )
    if idx_local is None:
        return None
    return int(idx_gate[idx_local])


def rebase_time_to_reference(
    t_s : np.ndarray,
    ref_y : np.ndarray,
    *,
    frac : float = 0.5,
    mode : Literal["rise", "fall", "abs"] = "rise",
    min_run : int = 2,
) -> Tuple[np.ndarray, Optional[int]]:
    idx_ref = estimate_edge_index(ref_y, frac=frac, t_s=t_s, mode=mode, min_run=min_run)
    t_s = np.asarray(t_s, dtype=np.float64)
    if idx_ref is None:
        return t_s.copy(), None
    return t_s - t_s[idx_ref], idx_ref