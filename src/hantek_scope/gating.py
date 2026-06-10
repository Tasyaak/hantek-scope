import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TimeGate:
    t1_s : float
    t2_s : float
    name : str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.t1_s) or not np.isfinite(self.t2_s):
            raise ValueError("t1_s and t2_s must be finite")
        if not (self.t1_s < self.t2_s):
            raise ValueError(f"Expected t1_s < t2_s, got {self.t1_s} >= {self.t2_s}")

    @property
    def width_s(self) -> float:
        return float(self.t2_s - self.t1_s)

    @property
    def label_us(self) -> str:
        return f"{self.t1_s * 1e6:.1f}–{self.t2_s * 1e6:.1f} мкс"


def gate_mask(t_s : np.ndarray, gate : TimeGate) -> np.ndarray:
    t_s = np.asarray(t_s, dtype=np.float64)
    return (t_s >= gate.t1_s) & (t_s <= gate.t2_s)


def gate_indices(t_s : np.ndarray, gate : TimeGate) -> np.ndarray:
    return np.flatnonzero(gate_mask(t_s, gate))


def extract_in_gate(t_s : np.ndarray, y : np.ndarray, gate : TimeGate) -> Tuple[np.ndarray, np.ndarray]:
    t_s = np.asarray(t_s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = gate_indices(t_s, gate)
    return t_s[idx], y[idx]


def peak_abs_in_gate(t_s : np.ndarray, y : np.ndarray, gate : TimeGate) -> float:
    _, seg = extract_in_gate(t_s, y, gate)
    if seg.size == 0:
        return float("nan")
    return float(np.max(np.abs(seg)))


def rms_in_gate(t_s : np.ndarray, y : np.ndarray, gate : TimeGate) -> float:
    _, seg = extract_in_gate(t_s, y, gate)
    if seg.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(seg ** 2)))


def rms_ac_in_gate(t_s : np.ndarray, y : np.ndarray, gate : TimeGate) -> float:
    _, seg = extract_in_gate(t_s, y, gate)
    if seg.size == 0:
        return float("nan")
    seg = seg - np.mean(seg)
    return float(np.sqrt(np.mean(seg ** 2)))


def argmax_abs_index_in_gate(t_s : np.ndarray, y : np.ndarray, gate : TimeGate) -> Optional[int]:
    t_s = np.asarray(t_s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = gate_indices(t_s, gate)
    if idx.size == 0:
        return None
    i_local = int(np.argmax(np.abs(y[idx])))
    return int(idx[i_local])


def crop_to_gate(t_s : np.ndarray, y : np.ndarray, gate : TimeGate) -> Tuple[np.ndarray, np.ndarray]:
    return extract_in_gate(t_s, y, gate)