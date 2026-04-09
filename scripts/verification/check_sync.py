import time, numpy as np, matplotlib.pyplot as plt
from hantek_scope.constants import MAX_DATA
from hantek_scope.calibration import raw_to_centered_codes
from hantek_scope.acquisition import estimate_ref_index_in_gate
from hantek_scope.averaging import _sleep_to_next_deadline
from hantek_scope import (
    HantekHardDll,
    ScanParams,
    DDSParams,
    DDSWaveType,
    DDSMode,
    StartControl,
    get_default_dll_dir,
    capture_average_burst_fast,
    TimeGate,
    extract_in_gate,
    peak_abs_in_gate,
    rms_ac_in_gate,
    argmax_abs_index_in_gate,
    make_scan_params_nominal,
)


DLL_DIR = get_default_dll_dir()

# -----------------------------
# Настройки эксперимента
# -----------------------------

K_SINGLE = 200          # сколько одиночных burst-кадров собрать для статистики джиттера
K_AVG = 5000            # сколько кадров усреднять в capture_average_burst_fast
REPEAT_S = 0.013        # период повторения burst
ARM_SETTLE_S = 0.001
RETRIES = 2
EDGE_FRAC = 0.5
OUTPUT_MODE = "codes"
REF_EDGE_MIN_RUN = 2

REF_GATE = TimeGate(t1_s=-3e-6, t2_s=10e-6, name="ref_gate")
PROMPT_GATE = TimeGate(t1_s=-2e-6, t2_s=8e-6, name="prompt_gate")
RX_GATE = TimeGate(t1_s=6e-6, t2_s=300e-6, name="rx_gate")
NOISE_GATE = TimeGate(t1_s=-40e-6, t2_s=-2e-6, name="noise_gate")

# Ограничение на lag при template matching single-burst -> averaged template
XCORR_MAX_LAG = 32  # samples, при dt=20 ns это ~0.64 us

# Пороги для point-level verdict
REF_STD_NEED_REBASING = 1.0
REF_SPAN90_NEED_REBASING = 2.0       # q95-q05 по ref edge index
REF_WIDTH_GAIN_REQUIRED = 0.95       # always должен сузить фронт хотя бы на 5%
RX_GAIN_REQUIRED = 1.05              # минимум +5% по RMS_ac/contrast
TEMPLATE_PEAK_GAIN_REQUIRED = 0.02   # прибавка к mean max normalized xcorr
TEMPLATE_LAG_SPAN_GAIN_REQUIRED = 0.75
PROMPT_RATIO_WORSE_LIMIT = 1.10


SCAN = make_scan_params_nominal(
    time_div="5us",
    read_len=0x8000,
    pretrigger_percent=10,
    rx_volt_div="50mV",
    rx_lever_pos=128,
    ref_volt_div="1V",
    ref_lever_pos=128,
    start_control=StartControl.NONE
)

DDS = DDSParams(
    frequency_hz=100_000.0,
    amplitude_mv=3500,
    offset_mv=0,
    wave_type=DDSWaveType.SQUARE,
    duty=0.25,
    burst_cycles=1,
    mode=DDSMode.BURST,
)


def pretrigger_mean(x: np.ndarray, t_s: np.ndarray) -> float:
    pre = t_s < 0.0
    if np.any(pre):
        return float(np.mean(x[pre]))
    n = max(1, x.size // 10)
    return float(np.mean(x[:n]))


def nominal_zero_from_lever(lever_pos: int) -> float:
    return float(MAX_DATA - lever_pos)


def to_centered_codes(raw_u16: np.ndarray, lever_pos: int) -> np.ndarray:
    return raw_to_centered_codes(raw_u16, zero_code_abs=nominal_zero_from_lever(lever_pos))


def rise_width_samples(y: np.ndarray) -> float:
    """
    Простейшая метрика "резкости" фронта:
    число отсчётов между 10% и 90% от положительного пика
    Чем меньше, тем фронт sharp-er
    """
    y = np.asarray(y, dtype=np.float64)
    n_head = max(1, y.size // 10)
    baseline = float(np.mean(y[:n_head]))
    yc = y - baseline
    peak = float(np.max(yc))
    if peak <= 0.0:
        return float("nan")

    idx10 = np.flatnonzero(yc >= 0.1 * peak)
    idx90 = np.flatnonzero(yc >= 0.9 * peak)
    if idx10.size == 0 or idx90.size == 0:
        return float("nan")
    return float(idx90[0] - idx10[0])


def require_array(name: str, x: np.ndarray | None) -> np.ndarray:
    if x is None:
        raise RuntimeError(f"{name} is None")
    return np.asarray(x, dtype=np.float64)


def plot_gate(ax, gate: TimeGate, *, color: str = "grey", alpha: float = 0.12) -> None:
    ax.axvspan(gate.t1_s * 1e6, gate.t2_s * 1e6, color=color, alpha=alpha)
    if gate.name:
        xc = 0.5 * (gate.t1_s + gate.t2_s) * 1e6
        ax.text(xc, 0.98, gate.name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)


def print_capture_window(scope : HantekHardDll, params : ScanParams) -> None:
    T_total = params.read_len / scope.fs_hz
    T_pre = params.pretrigger_percent / 100.0 * T_total
    T_post = T_total - T_pre
    print(f"K_AVG={K_AVG}")
    print(f"time_div={params.time_div}")
    print(f"rx_volt_div={params.rx_volt_div}")
    print(f"frequency_hz={DDS.frequency_hz} Hz")
    print(f"amplitude_mv={DDS.amplitude_mv} mV")
    print(f"duty={DDS.duty}\n")
    print(f"fs_hz={scope.fs_hz:.6e} Hz")
    print(f"dt_s={scope.dt_s:.6e} s")
    print(f"read_len={params.read_len}")
    print(f"T_total={T_total*1e6:.3f} us")
    print(f"T_pre={T_pre*1e6:.3f} us")
    print(f"T_post={T_post*1e6:.3f} us")


def safe_ratio(a: float, b: float, eps: float = 1e-15) -> float:
    return float(a / max(abs(b), eps))


def robust_stats(x: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "mad": float("nan"),
            "q05": float("nan"),
            "q95": float("nan"),
            "span90": float("nan"),
        }

    med = float(np.median(arr))
    q05, q95 = np.quantile(arr, [0.05, 0.95])
    mad = float(np.median(np.abs(arr - med)))

    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": med,
        "mad": mad,
        "q05": float(q05),
        "q95": float(q95),
        "span90": float(q95 - q05),
    }


def print_robust_stats(name: str, s: dict[str, float], suffix: str = "") -> None:
    print(
        f"{name:<28}: "
        f"mean={s['mean']:.6f}{suffix}, "
        f"std={s['std']:.6f}{suffix}, "
        f"median={s['median']:.6f}{suffix}, "
        f"MAD={s['mad']:.6f}{suffix}, "
        f"q05={s['q05']:.6f}{suffix}, "
        f"q95={s['q95']:.6f}{suffix}, "
        f"span90={s['span90']:.6f}{suffix}"
    )


def max_norm_xcorr_and_lag(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag: int | None = None,
) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 1e-15 or ny <= 1e-15:
        return float("nan"), 0

    cc = np.correlate(x / nx, y / ny, mode="full")
    lags = np.arange(-(len(y) - 1), len(x), dtype=np.int64)

    if max_lag is not None:
        mask = np.abs(lags) <= int(max_lag)
        cc = cc[mask]
        lags = lags[mask]

    i = int(np.argmax(cc))
    return float(cc[i]), int(lags[i])


def template_match_stats(
    traces: list[np.ndarray],
    template: np.ndarray,
    *,
    max_lag: int,
) -> dict[str, float]:
    peak_vals: list[float] = []
    lag_vals: list[float] = []

    for tr in traces:
        peak, lag = max_norm_xcorr_and_lag(tr, template, max_lag=max_lag)
        if np.isfinite(peak):
            peak_vals.append(float(peak))
            lag_vals.append(float(lag))

    s_peak = robust_stats(peak_vals)
    s_lag = robust_stats(lag_vals)

    return {
        "peak_mean": s_peak["mean"],
        "peak_std": s_peak["std"],
        "peak_median": s_peak["median"],
        "peak_mad": s_peak["mad"],
        "peak_q05": s_peak["q05"],
        "peak_q95": s_peak["q95"],
        "peak_span90": s_peak["span90"],

        "lag_mean": s_lag["mean"],
        "lag_std": s_lag["std"],
        "lag_median": s_lag["median"],
        "lag_mad": s_lag["mad"],
        "lag_q05": s_lag["q05"],
        "lag_q95": s_lag["q95"],
        "lag_span90": s_lag["span90"],
    }


def print_template_stats(name: str, s: dict[str, float]) -> None:
    print(
        f"{name:<28}: "
        f"xcorr_mean={s['peak_mean']:.6f}, "
        f"xcorr_std={s['peak_std']:.6f}, "
        f"xcorr_q05={s['peak_q05']:.6f}, "
        f"xcorr_q95={s['peak_q95']:.6f}, "
        f"lag_median={s['lag_median']:.3f} samples, "
        f"lag_MAD={s['lag_mad']:.3f} samples, "
        f"lag_span90={s['lag_span90']:.3f} samples"
    )


def main() -> None:
    with HantekHardDll(DLL_DIR) as scope:
        scope.configure(SCAN)
        print_capture_window(scope, SCAN)
        scope.dds_configure(DDS, force=True)
        scope.dds_output(True)
        time.sleep(0.2)

        # -------------------------------------------------------
        # 1) Собираем K_SINGLE одиночных burst-кадров и меряем
        #    разброс ref edge index на CH2
        # -------------------------------------------------------
        
        ref_edge_idxs_arr : list[int] = []
        rx_peak_idxs_arr : list[int] = []
        t_axis : np.ndarray | None = None
        ref_traces : list[np.ndarray] = []
        rx_traces : list[np.ndarray] = []

        rx_gate_single_traces: list[np.ndarray] = []
        prompt_peak_single_arr: list[float] = []
        noise_rms_single_arr: list[float] = []

        next_deadline = time.perf_counter()

        skipped_untriggered = 0
        skipped_overflow = 0
        skipped_missing_ref_edge = 0
        skipped_missing_rx_peak = 0

        for _ in range(K_SINGLE):
            fr = scope.capture_burst(
                return_mode="raw",
                copy=True,
                apply_vert_cal=False,
                arm_settle_s=ARM_SETTLE_S,
                retries=RETRIES,
            )

            if not fr.triggered:
                skipped_untriggered += 1
                next_deadline = _sleep_to_next_deadline(next_deadline, REPEAT_S)
                continue

            if fr.rx_overflow or fr.ref_overflow:
                skipped_overflow += 1
                next_deadline = _sleep_to_next_deadline(next_deadline, REPEAT_S)
                continue

            assert fr.rx_raw_u16 is not None
            assert fr.ref_raw_u16 is not None

            rx_codes = to_centered_codes(fr.rx_raw_u16, SCAN.rx_lever_pos)
            ref_codes = to_centered_codes(fr.ref_raw_u16, SCAN.ref_lever_pos)

            rx_codes = rx_codes - pretrigger_mean(rx_codes, fr.t_s)
            ref_codes = ref_codes - pretrigger_mean(ref_codes, fr.t_s)

            idx_ref = estimate_ref_index_in_gate(
                fr.t_s,
                ref_codes,
                REF_GATE,
                frac=EDGE_FRAC,
                pretrigger_percent=SCAN.pretrigger_percent,
                mode="rise",
                min_run=REF_EDGE_MIN_RUN,
            )

            if idx_ref is None:
                skipped_missing_ref_edge += 1
                next_deadline = _sleep_to_next_deadline(next_deadline, REPEAT_S)
                continue

            idx_rx = argmax_abs_index_in_gate(fr.t_s, rx_codes, RX_GATE)
            if idx_rx is None:
                skipped_missing_rx_peak += 1
                next_deadline = _sleep_to_next_deadline(next_deadline, REPEAT_S)
                continue

            ref_edge_idxs_arr.append(idx_ref)
            rx_peak_idxs_arr.append(idx_rx)

            _, rx_gate_seg = extract_in_gate(fr.t_s, rx_codes, RX_GATE)
            if rx_gate_seg.size > 0:
                rx_gate_single_traces.append(np.asarray(rx_gate_seg, dtype=np.float32).copy())

            prompt_peak_single_arr.append(peak_abs_in_gate(fr.t_s, rx_codes, PROMPT_GATE))
            noise_rms_single_arr.append(rms_ac_in_gate(fr.t_s, rx_codes, NOISE_GATE))

            if t_axis is None:
                t_axis = np.asarray(fr.t_s, dtype=np.float64).copy()

            # сохраним не все, а только первые 30 трасс для графиков
            if len(ref_traces) < 30:
                ref_traces.append(ref_codes.copy())
                rx_traces.append(rx_codes.copy())

            next_deadline = _sleep_to_next_deadline(next_deadline, REPEAT_S)

        ref_edge_idxs_arr = np.asarray(ref_edge_idxs_arr, dtype=np.float64) # type: ignore
        rx_peak_idxs_arr = np.asarray(rx_peak_idxs_arr, dtype=np.float64) # type: ignore

        print("=== SINGLE BURST STATS ===")
        print(f"valid frames               : {ref_edge_idxs_arr.size}") # type: ignore
        print(f"skipped_untriggered        : {skipped_untriggered}")
        print(f"skipped_overflow           : {skipped_overflow}")
        print(f"skipped_missing_ref_edge   : {skipped_missing_ref_edge}")
        print(f"skipped_missing_rx_peak    : {skipped_missing_rx_peak}")

        if ref_edge_idxs_arr.size == 0: # type: ignore
            raise RuntimeError("No valid single-burst frames for synchronization analysis")

        print(f"ref edge mean [samples]    : {np.mean(ref_edge_idxs_arr):.3f}")
        print(f"ref edge std  [samples]    : {np.std(ref_edge_idxs_arr, ddof=0):.3f}")
        print(f"ref edge min/max [samples] : {np.min(ref_edge_idxs_arr):.0f} / {np.max(ref_edge_idxs_arr):.0f}")

        if rx_peak_idxs_arr.size > 0: # type: ignore
            print(f"rx peak mean [samples]     : {np.mean(rx_peak_idxs_arr):.3f}")
            print(f"rx peak std  [samples]     : {np.std(rx_peak_idxs_arr, ddof=0):.3f}")

        ref_edge_stats = robust_stats(ref_edge_idxs_arr) # type: ignore
        rx_peak_stats = robust_stats(rx_peak_idxs_arr) # type: ignore
        prompt_peak_single_stats = robust_stats(prompt_peak_single_arr)
        noise_rms_single_stats = robust_stats(noise_rms_single_arr)

        print("\n=== SINGLE BURST ROBUST STATS ===")
        print_robust_stats("ref edge index", ref_edge_stats, " samples")
        print_robust_stats("rx argmax index", rx_peak_stats, " samples")
        print_robust_stats("prompt |peak|", prompt_peak_single_stats, " codes")
        print_robust_stats("noise RMS_ac", noise_rms_single_stats, " codes")

        # -------------------------------------------------------
        # 2) Сравниваем среднее без rebasing и с rebasing
        # -------------------------------------------------------

        res_never = capture_average_burst_fast(
            scope,
            K=K_AVG,
            repeat_s=REPEAT_S,
            edge_frac=EDGE_FRAC,
            rebasing="never",
            pilot_frames=32,
            jitter_threshold_samples=1.0,
            arm_settle_s=ARM_SETTLE_S,
            retries=RETRIES,
            apply_vert_cal=False,
            baseline_mode="rx_pretrigger",
            skip_overflow=True,
            skip_untriggered=True,
            skip_missing_ref_edge=True,
            output_mode=OUTPUT_MODE,
            max_attempt_factor=3,
            return_ref=True,
            ref_edge_gate=REF_GATE,
            ref_edge_mode="rise",
            ref_edge_min_run=REF_EDGE_MIN_RUN,
            return_baseline_stats=True,
        )

        res_always = capture_average_burst_fast(
            scope,
            K=K_AVG,
            repeat_s=REPEAT_S,
            edge_frac=EDGE_FRAC,
            rebasing="always",
            pilot_frames=32,
            jitter_threshold_samples=1.0,
            arm_settle_s=ARM_SETTLE_S,
            retries=RETRIES,
            apply_vert_cal=False,
            baseline_mode="rx_pretrigger",
            skip_overflow=True,
            skip_untriggered=True,
            skip_missing_ref_edge=True,
            output_mode=OUTPUT_MODE,
            max_attempt_factor=3,
            return_ref=True,
            ref_edge_gate=REF_GATE,
            ref_edge_mode="rise",
            ref_edge_min_run=REF_EDGE_MIN_RUN,
            return_baseline_stats=True,
        )

        ref_never = require_array("res_never.ref_codes", res_never.ref_codes)
        ref_always = require_array("res_always.ref_codes", res_always.ref_codes)
        rx_never = require_array("res_never.rx_codes", res_never.rx_codes)
        rx_always = require_array("res_always.rx_codes", res_always.rx_codes)

        # CH2 reference: меряем качество только в REF_GATE
        _, ref_never_gate = extract_in_gate(res_never.t_rebased_s, ref_never, REF_GATE)
        _, ref_always_gate = extract_in_gate(res_always.t_rebased_s, ref_always, REF_GATE)
        
        ref_width_never = rise_width_samples(ref_never_gate) if ref_never_gate.size else float("nan")
        ref_width_always = rise_width_samples(ref_always_gate) if ref_always_gate.size else float("nan")

        # CH1 receiver: метрики только в RX_GATE
        rx_peak_never = peak_abs_in_gate(res_never.t_rebased_s, rx_never, RX_GATE)
        rx_peak_always = peak_abs_in_gate(res_always.t_rebased_s, rx_always, RX_GATE)

        rx_rms_never = rms_ac_in_gate(res_never.t_rebased_s, rx_never, RX_GATE)
        rx_rms_always = rms_ac_in_gate(res_always.t_rebased_s, rx_always, RX_GATE)

        noise_rms_never = rms_ac_in_gate(res_never.t_rebased_s, rx_never, NOISE_GATE)
        noise_rms_always = rms_ac_in_gate(res_always.t_rebased_s, rx_always, NOISE_GATE)

        contrast_never = rx_rms_never / max(noise_rms_never, 1e-15)
        contrast_always = rx_rms_always / max(noise_rms_always, 1e-15)

        prompt_peak_never = peak_abs_in_gate(res_never.t_rebased_s, rx_never, PROMPT_GATE)
        prompt_peak_always = peak_abs_in_gate(res_always.t_rebased_s, rx_always, PROMPT_GATE)

        prompt_to_rx_never = safe_ratio(prompt_peak_never, rx_peak_never)
        prompt_to_rx_always = safe_ratio(prompt_peak_always, rx_peak_always)

        # _, tpl_never = extract_in_gate(res_never.t_rebased_s, rx_never, RX_GATE)
        # _, tpl_always = extract_in_gate(res_always.t_rebased_s, rx_always, RX_GATE)

        # tmpl_match_never = template_match_stats(
        #     rx_gate_single_traces,
        #     tpl_never,
        #     max_lag=XCORR_MAX_LAG,
        # )
        # tmpl_match_always = template_match_stats(
        #     rx_gate_single_traces,
        #     tpl_always,
        #     max_lag=XCORR_MAX_LAG,
        # )

        print("\n=== AVERAGED RESULT COMPARISON ===")
        print(f"never.used_rebasing        : {res_never.used_rebasing}")
        print(f"always.used_rebasing       : {res_always.used_rebasing}")
        print(f"always.ref_edge_std_samples: {res_always.ref_edge_std_samples:.3f}")
        print(f"ref rise width NEVER       : {ref_width_never:.3f} samples")
        print(f"ref rise width ALWAYS      : {ref_width_always:.3f} samples")
        print(f"rx |peak| NEVER            : {rx_peak_never:.6f} codes")
        print(f"rx |peak| ALWAYS           : {rx_peak_always:.6f} codes")
        print(f"peak gain ALWAYS/NEVER     : {rx_peak_always / max(rx_peak_never, 1e-15):.3f}")
        print(f"rx RMS_ac NEVER            : {rx_rms_never:.6f} codes")
        print(f"rx RMS_ac ALWAYS           : {rx_rms_always:.6f} codes")
        print(f"contrast NEVER             : {contrast_never:.3f}")
        print(f"contrast ALWAYS            : {contrast_always:.3f}")
        print(f"rx baseline mean NEVER     : {res_never.rx_baseline_mean_codes:.6f} codes")
        print(f"rx baseline std  NEVER     : {res_never.rx_baseline_std_codes:.6f} codes")
        print(f"rx baseline mean ALWAYS    : {res_always.rx_baseline_mean_codes:.6f} codes")
        print(f"rx baseline std  ALWAYS    : {res_always.rx_baseline_std_codes:.6f} codes")

        assert res_never.rx_baseline_std_codes is not None
        assert res_always.rx_baseline_std_codes is not None

        baseline_drift_ratio_never = res_never.rx_baseline_std_codes / max(noise_rms_never, 1e-15)
        baseline_drift_ratio_always = res_always.rx_baseline_std_codes / max(noise_rms_always, 1e-15)

        print(f"baseline/noise NEVER       : {baseline_drift_ratio_never:.3f}")
        print(f"baseline/noise ALWAYS      : {baseline_drift_ratio_always:.3f}")

        print(f"prompt |peak| NEVER        : {prompt_peak_never:.6f} codes")
        print(f"prompt |peak| ALWAYS       : {prompt_peak_always:.6f} codes")
        print(f"prompt/rx peak NEVER       : {prompt_to_rx_never:.6f}")
        print(f"prompt/rx peak ALWAYS      : {prompt_to_rx_always:.6f}")

        # print("\n=== SINGLE -> TEMPLATE MATCH (RX_GATE) ===")
        # print_template_stats("single -> template NEVER", tmpl_match_never)
        # print_template_stats("single -> template ALWAYS", tmpl_match_always)

        ref_jitter_material = (
            np.isfinite(res_always.ref_edge_std_samples)
            and (res_always.ref_edge_std_samples >= REF_STD_NEED_REBASING)
            and (ref_edge_stats["span90"] >= REF_SPAN90_NEED_REBASING)
        )

        ref_sharpens = (
            np.isfinite(ref_width_never)
            and np.isfinite(ref_width_always)
            and (ref_width_always <= REF_WIDTH_GAIN_REQUIRED * ref_width_never)
        )

        rx_improves = (
            (rx_rms_always >= RX_GAIN_REQUIRED * rx_rms_never)
            or (contrast_always >= RX_GAIN_REQUIRED * contrast_never)
        )

        # template_improves = (
        #     np.isfinite(tmpl_match_never["peak_mean"])
        #     and np.isfinite(tmpl_match_always["peak_mean"])
        #     and np.isfinite(tmpl_match_never["lag_span90"])
        #     and np.isfinite(tmpl_match_always["lag_span90"])
        #     and (tmpl_match_always["peak_mean"] >= tmpl_match_never["peak_mean"] + TEMPLATE_PEAK_GAIN_REQUIRED)
        #     and (tmpl_match_always["lag_span90"] <= TEMPLATE_LAG_SPAN_GAIN_REQUIRED * tmpl_match_never["lag_span90"])
        # )

        prompt_not_worse = (
            np.isfinite(prompt_to_rx_never)
            and np.isfinite(prompt_to_rx_always)
            and (prompt_to_rx_always <= PROMPT_RATIO_WORSE_LIMIT * prompt_to_rx_never)
        )

        # need_rebasing_point = (
        #     ref_jitter_material
        #     and ref_sharpens
        #     and rx_improves
        #     and template_improves
        #     and prompt_not_worse
        # )

        print("\n=== POINT-LEVEL VERDICT ===")
        print(f"ref_jitter_material        : {ref_jitter_material}")
        print(f"ref_sharpens               : {ref_sharpens}")
        print(f"rx_improves                : {rx_improves}")
        # print(f"template_improves          : {template_improves}")
        print(f"prompt_not_worse           : {prompt_not_worse}")

        # if need_rebasing_point:
        #     print("Point verdict: rebasing IS justified at this point")
        # else:
        #     print("Point verdict: rebasing is NOT justified at this point")

        print(
            "\nGrid-level note: this verdict is valid only for the current point. "
            "To make a configuration-level decision for the whole grid, repeat the same "
            "script at several representative points: near the piezo, mid-range, far corner, "
            "near an edge, and over/near a suspected defect region."
        )

        # -------------------------------------------------------
        # 3) Графики
        # -------------------------------------------------------
        if t_axis is None:
            raise RuntimeError("t_axis is None")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        ax = axes[0, 0]
        ax.hist(ref_edge_idxs_arr, bins=min(40, max(10, ref_edge_idxs_arr.size // 5))) # type: ignore
        ax.set_title("Histogram of CH2 reference edge index")
        ax.set_xlabel("sample index")
        ax.set_ylabel("count")

        ax = axes[0, 1]
        for tr in ref_traces:
            ax.plot(t_axis * 1e6, tr, alpha=0.35)
        plot_gate(ax, REF_GATE, color="tab:green", alpha=0.15)
        ax.set_title("Single-burst CH2 traces (centered codes)")
        ax.set_xlabel("time, us")
        ax.set_ylabel("codes")

        ax = axes[1, 0]
        ax.plot(res_never.t_rebased_s * 1e6, ref_never, label="avg ref, rebasing=never")
        ax.plot(res_always.t_rebased_s * 1e6, ref_always, label="avg ref, rebasing=always")
        plot_gate(ax, REF_GATE, color="tab:green", alpha=0.15)
        ax.set_title("Averaged CH2 reference")
        ax.set_xlabel("time, us")
        ax.set_ylabel("codes")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(res_never.t_rebased_s * 1e6, rx_never, label="avg rx, rebasing=never")
        ax.plot(res_always.t_rebased_s * 1e6, rx_always, label="avg rx, rebasing=always")
        plot_gate(ax, PROMPT_GATE, color="tab:red", alpha=0.08)
        plot_gate(ax, RX_GATE, color="tab:orange", alpha=0.12)
        plot_gate(ax, NOISE_GATE, color="tab:purple", alpha=0.10)
        ax.set_title("Averaged CH1 receiver")
        ax.set_xlabel("time, us")
        ax.set_ylabel("codes")
        ax.legend()

        fig.tight_layout()

        # fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

        # lags_never = []
        # peaks_never = []
        # lags_always = []
        # peaks_always = []

        # for tr in rx_gate_single_traces:
        #     p_n, l_n = max_norm_xcorr_and_lag(tr, tpl_never, max_lag=XCORR_MAX_LAG)
        #     p_a, l_a = max_norm_xcorr_and_lag(tr, tpl_always, max_lag=XCORR_MAX_LAG)
        #     if np.isfinite(p_n):
        #         peaks_never.append(p_n)
        #         lags_never.append(l_n)
        #     if np.isfinite(p_a):
        #         peaks_always.append(p_a)
        #         lags_always.append(l_a)

        # axes2[0].hist(lags_never, alpha=0.6, label="never template")
        # axes2[0].hist(lags_always, alpha=0.6, label="always template")
        # axes2[0].set_title("Single->template lag histogram")
        # axes2[0].set_xlabel("lag, samples")
        # axes2[0].set_ylabel("count")
        # axes2[0].legend()

        # axes2[1].hist(peaks_never, alpha=0.6, label="never template")
        # axes2[1].hist(peaks_always, alpha=0.6, label="always template")
        # axes2[1].set_title("Single->template max normalized xcorr")
        # axes2[1].set_xlabel("peak xcorr")
        # axes2[1].set_ylabel("count")
        # axes2[1].legend()

        # fig2.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()