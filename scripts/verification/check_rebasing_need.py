import numpy as np, time
from typing import Literal, TypedDict
from hantek_scope import (
    HantekHardDll, StartControl, make_scan_params_nominal,
    DDSParams, DDSWaveType, DDSMode,
    capture_average_burst_fast, get_default_dll_dir,
    TimeGate, peak_abs_in_gate, rms_ac_in_gate,
)


DLL_DIR = get_default_dll_dir()

K = 5000
REPEAT_S = 0.013
N_RUNS = 5

EDGE_STD_THRESHOLD_SAMPLES = 1.0
MIN_ALWAYS_GAIN = 1.03
AUTO_USE_RATIO_THRESHOLD = 0.6
OUTPUT_MODE : Literal["codes", "nominal_volts"] = "codes"

REF_GATE = TimeGate(t1_s=-3e-6, t2_s=10e-6, name="ref_gate")
RX_GATE = TimeGate(t1_s=7e-6, t2_s=300e-6, name="rx_gate")
TIME_AXIS_MODE : Literal["t_s", "t_rebased_s"] = "t_rebased_s"

class RunStats(TypedDict):
    used_rebasing: bool
    ref_edge_std_samples: float
    peak_abs: float
    rms_ac: float


def peak_abs(y: np.ndarray) -> float:
    return float(np.max(np.abs(y)))


def rms_ac(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    return float(np.sqrt(np.mean(y ** 2)))


def require_array(name: str, x: np.ndarray | None) -> np.ndarray:
    if x is None:
        raise RuntimeError(f"{name} is None")
    return np.asarray(x, dtype=np.float64)


def value_units() -> str:
    return "codes" if OUTPUT_MODE == "codes" else "nominal V"


def get_rx_array(res) -> np.ndarray:
    if OUTPUT_MODE == "codes":
        return require_array("res.rx_codes", res.rx_codes)
    return require_array("res.rx_volts", res.rx_volts)


def get_time_array(res) -> np.ndarray:
    if TIME_AXIS_MODE == "t_rebased_s":
        return np.asarray(res.t_rebased_s, dtype=np.float64)
    return np.asarray(res.t_s, dtype=np.float64)


def finite_stats(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), int(arr.size)


def fmt_stat(x: float, suffix: str = "") -> str:
    if np.isfinite(x):
        return f"{x:.6f}{suffix}"
    return f"n/a{suffix}"


def summarize_results(name: str, results: list[RunStats]) -> None:
    peak_vals = [r["peak_abs"] for r in results]
    rms_vals = [r["rms_ac"] for r in results]
    edge_std_vals = [r["ref_edge_std_samples"] for r in results]
    used_flags = [r["used_rebasing"] for r in results]
    units = value_units()

    edge_mean, edge_std, edge_n = finite_stats(edge_std_vals)
    peak_mean, peak_std, _ = finite_stats(peak_vals)
    rms_mean, rms_std, _ = finite_stats(rms_vals)

    print(f"\n=== {name} ===")
    print(f"runs                      = {len(results)}")
    print(f"used_rebasing count       = {sum(1 for x in used_flags if x)} / {len(used_flags)}")
    print(f"ref_edge_std finite count = {edge_n} / {len(edge_std_vals)}")
    print(f"ref_edge_std mean         = {fmt_stat(edge_mean, ' samples')}")
    print(f"ref_edge_std std          = {fmt_stat(edge_std, ' samples')}")
    print(f"peak_abs mean             = {fmt_stat(peak_mean, f' {units}')}")
    print(f"peak_abs std              = {fmt_stat(peak_std, f' {units}')}")
    print(f"rms_ac mean               = {fmt_stat(rms_mean, f' {units}')}")
    print(f"rms_ac std                = {fmt_stat(rms_std, f' {units}')}")


def run_mode(scope: HantekHardDll, mode: Literal["never", "auto", "always"]) -> list[RunStats]:
    out : list[RunStats] = []

    for _ in range(N_RUNS):
        res = capture_average_burst_fast(
            scope,
            K=K,
            repeat_s=REPEAT_S,
            rebasing=mode,
            pilot_frames=32,
            jitter_threshold_samples=EDGE_STD_THRESHOLD_SAMPLES,
            arm_settle_s=0.001,
            retries=2,
            output_mode=OUTPUT_MODE,
            return_ref=True,
            ref_edge_gate=REF_GATE,
            ref_edge_mode="rise",
            ref_edge_min_run=3,
            return_baseline_stats=True,
        )

        rx = get_rx_array(res)
        t = get_time_array(res)

        out.append({
            "used_rebasing": bool(res.used_rebasing),
            "ref_edge_std_samples": float(res.ref_edge_std_samples),
            "peak_abs": peak_abs_in_gate(t, rx, RX_GATE),
            "rms_ac": rms_ac_in_gate(t, rx, RX_GATE),
        })
    return out


def main() -> None:
    scan = make_scan_params_nominal(
        time_div="5us",
        read_len=0x8000,
        pretrigger_percent=10,
        rx_volt_div="100mV",
        rx_lever_pos=128,
        ref_volt_div="1V",
        ref_lever_pos=128,
        start_control=StartControl.NONE
    )
    
    dds = DDSParams(
        frequency_hz=125_000.0,
        amplitude_mv=3500,
        offset_mv=0,
        wave_type=DDSWaveType.SQUARE,
        phase_frac=0.0,
        burst_cycles=1,
        mode=DDSMode.BURST,
        duty=0.3,
    )

    with HantekHardDll(DLL_DIR, channel_mode=4) as scope:
        scope.configure(scan)
        scope.dds_configure(dds, force=True)
        scope.dds_output(True)
        time.sleep(0.1)

        try:
            results_never = run_mode(scope, "never")
            results_auto = run_mode(scope, "auto")
            results_always = run_mode(scope, "always")
        finally:
            scope.dds_stop()

    summarize_results("NEVER", results_never)
    summarize_results("AUTO", results_auto)
    summarize_results("ALWAYS", results_always)

    peak_never = float(np.mean([r["peak_abs"] for r in results_never]))
    peak_auto = float(np.mean([r["peak_abs"] for r in results_auto]))
    peak_always = float(np.mean([r["peak_abs"] for r in results_always]))

    rms_never = float(np.mean([r["rms_ac"] for r in results_never]))
    rms_auto = float(np.mean([r["rms_ac"] for r in results_auto]))
    rms_always = float(np.mean([r["rms_ac"] for r in results_always]))

    edge_auto_vals = np.asarray([r["ref_edge_std_samples"] for r in results_auto], dtype=np.float64)
    edge_auto_finite = edge_auto_vals[np.isfinite(edge_auto_vals)]
    edge_auto = float(np.mean(edge_auto_finite)) if edge_auto_finite.size else float("nan")

    used_auto_count = sum(1 for r in results_auto if r["used_rebasing"])
    auto_use_ratio = used_auto_count / max(N_RUNS, 1)

    peak_gain_auto = peak_auto / max(peak_never, 1e-15)
    rms_gain_auto = rms_auto / max(rms_never, 1e-15)

    peak_gain_always = peak_always / max(peak_never, 1e-15)
    rms_gain_always = rms_always / max(rms_never, 1e-15)
    auto_use_ratio = used_auto_count / max(N_RUNS, 1)

    print("\n=== COMPARISON ===")
    print(f"peak gain AUTO   vs NEVER  = {peak_gain_auto:.6f}")
    print(f"rms gain  AUTO   vs NEVER  = {rms_gain_auto:.6f}")
    print(f"peak gain ALWAYS vs NEVER  = {peak_gain_always:.6f}")
    print(f"rms gain  ALWAYS vs NEVER  = {rms_gain_always:.6f}")
    print(f"AUTO used rebasing         = {used_auto_count}/{N_RUNS}")
    print(f"AUTO use ratio             = {auto_use_ratio:.3f}")
    print(f"AUTO mean ref_edge_std     = {fmt_stat(edge_auto, ' samples')}")

    need_rebasing = (
        auto_use_ratio >= AUTO_USE_RATIO_THRESHOLD
        or (np.isfinite(edge_auto) and edge_auto > EDGE_STD_THRESHOLD_SAMPLES)
        or peak_gain_always >= MIN_ALWAYS_GAIN
        or rms_gain_always >= MIN_ALWAYS_GAIN
    )

    if not need_rebasing:
        print("\nRecommendation: rebasing is NOT necessary for this configuration")
    else:
        print("\nRecommendation: use rebasing for this configuration")


if __name__ == "__main__":
    main()