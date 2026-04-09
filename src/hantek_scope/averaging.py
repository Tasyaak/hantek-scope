import numpy as np, time
from typing import Literal, Tuple, Optional
from .device import HantekHardDll
from .models import AveragedBurstResult, ScanPointPayload
from .calibration import raw_to_centered_codes, codes_to_volts
from .acquisition import estimate_edge_index, estimate_ref_index_in_gate
from .gating import TimeGate


def accumulate_shifted(dst: np.ndarray, src: np.ndarray, weight: np.ndarray, shift: int) -> None:
    """
    shift > 0 : move src to the right
    shift < 0 : move src to the left
    """
    n = len(src)
    if shift == 0:
        dst += src
        weight += 1.0
        return

    if shift > 0:
        if shift < n:
            dst[shift:] += src[:n - shift]
            weight[shift:] += 1.0
        return

    # shift < 0
    s = -shift
    if s < n:
        dst[:n - s] += src[s:]
        weight[:n - s] += 1.0


def _estimate_pretrigger_mean(x: np.ndarray, t_s: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)

    pre = t_s < 0.0
    if np.any(pre):
        return float(np.mean(x[pre]))

    n = max(1, x.size // 10)
    return float(np.mean(x[:n]))


def _select_current_vertical_cal(scope: HantekHardDll, apply_vert_cal: bool) -> Tuple[float, float, float, float]:
    if apply_vert_cal:
        return (
            scope._rx_zero_code_abs,
            scope._ref_zero_code_abs,
            scope._rx_scale_v_per_code,
            scope._ref_scale_v_per_code,
        )
    return (
        scope._rx_nominal_zero_code_abs,
        scope._ref_nominal_zero_code_abs,
        scope._rx_nominal_scale_v_per_code,
        scope._ref_nominal_scale_v_per_code,
    )


def _sleep_to_next_deadline(next_deadline : float, repeat_s : float) -> float:
    next_deadline += repeat_s
    sleep_s = next_deadline - time.perf_counter()
    if sleep_s > 0:
        time.sleep(sleep_s)
    return next_deadline


def should_skip_for_overflow(*, rx_overflow : bool, ref_overflow : bool, need_ref_processing : bool) -> bool:
    if rx_overflow:
        return True
    if need_ref_processing and ref_overflow:
        return True
    return False


def capture_average_burst_fast(
    scope: HantekHardDll,
    *,
    K: int,
    repeat_s: float,
    edge_frac: float = 0.5,
    rebasing: Literal["auto", "always", "never"] = "never",
    pilot_frames: int = 32,
    jitter_threshold_samples: float = 1.0,
    arm_settle_s: float = 0.001,
    retries: int = 2,
    apply_vert_cal: bool = True,
    baseline_mode: Literal["none", "rx_pretrigger", "both_pretrigger"] = "rx_pretrigger",
    skip_overflow: bool = True,
    skip_untriggered: bool = True,
    skip_missing_ref_edge: bool = True,
    output_mode: Literal["codes", "nominal_volts", "calibrated_volts"] = "codes",
    max_attempt_factor: int = 3,
    return_ref: bool = True,
    ref_edge_gate: Optional[TimeGate] = None,
    ref_edge_mode: Literal["rise", "fall", "abs"] = "rise",
    ref_edge_min_run: int = 2,
    return_baseline_stats: bool = True,
) -> AveragedBurstResult:
    if K < 1:
        raise ValueError("K must be >= 1")
    if repeat_s <= 0:
        raise ValueError("repeat_s must be > 0")
    if max_attempt_factor < 1:
        raise ValueError("max_attempt_factor must be >= 1")
    if output_mode not in {"codes", "nominal_volts", "calibrated_volts"}:
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")

    p = scope.params
    if p is None:
        raise RuntimeError("scope must be configured first")
    if p.ref_channel is None:
        raise RuntimeError("reference channel is required for synchronized averaging")

    if output_mode == "calibrated_volts":
        apply_vert_cal_effective = True
    elif output_mode == "nominal_volts":
        apply_vert_cal_effective = False
    else:
        # для "codes" можно оставить старое поведение apply_vert_cal
        apply_vert_cal_effective = apply_vert_cal

    (
        rx_zero_code_abs,
        ref_zero_code_abs,
        rx_scale_v_per_code,
        ref_scale_v_per_code,
    ) = _select_current_vertical_cal(scope, apply_vert_cal=apply_vert_cal_effective)

    skipped_overflow = 0
    skipped_untriggered = 0
    skipped_missing_ref_edge = 0
    accepted_count = 0

    rx_baseline_codes_all : list[float] = []
    pilot_data : list[Tuple[np.ndarray, Optional[np.ndarray], Optional[int], float]] = []
    edge_idxs : list[int] = []

    # Нужен ли ref-edge в pilot-фазе:
    # - при rebasing="always" нужен
    # - при rebasing="auto" нужен
    # - при rebasing="never" не нужен
    need_pilot_ref_edge = (rebasing != "never")

    # Нужен ли вообще ref-processing в pilot-фазе:
    # - если нужно вернуть ref наружу
    # - или если нужен ref-edge
    need_pilot_ref_processing = return_ref or need_pilot_ref_edge

    target_pilot_count = min(K, max(1, pilot_frames if need_pilot_ref_edge else 1))

    max_attempts = max(K * max_attempt_factor, K + target_pilot_count + 8)
    attempts = 0
    t_s0 : Optional[np.ndarray] = None
    next_deadline = time.perf_counter()

    def preprocess_frame(
        fr,
        *,
        need_ref_processing: bool,
        need_ref_edge: bool,
        t_s: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int], float]:
        assert fr.rx_raw_u16 is not None

        rx_codes = raw_to_centered_codes(fr.rx_raw_u16, zero_code_abs=rx_zero_code_abs)

        rx_baseline_codes = 0.0
        if baseline_mode in {"rx_pretrigger", "both_pretrigger"}:
            rx_baseline_codes = _estimate_pretrigger_mean(rx_codes, t_s)
            rx_codes = rx_codes - rx_baseline_codes

        ref_codes : Optional[np.ndarray] = None
        idx_ref : Optional[int] = None

        if need_ref_processing:
            assert fr.ref_raw_u16 is not None
            ref_codes = raw_to_centered_codes(fr.ref_raw_u16, zero_code_abs=ref_zero_code_abs)

            if baseline_mode == "both_pretrigger":
                ref_baseline_codes = _estimate_pretrigger_mean(ref_codes, t_s)
                ref_codes = ref_codes - ref_baseline_codes

            if need_ref_edge:
                if ref_edge_gate is None:
                    idx_ref = estimate_edge_index(
                        ref_codes,
                        frac=edge_frac,
                        t_s=t_s,
                        baseline_head_fraction=max(0.01, p.pretrigger_percent / 100.0),
                        mode=ref_edge_mode,
                        min_run=ref_edge_min_run,
                    )
                else:
                    idx_ref = estimate_ref_index_in_gate(
                        t_s,
                        ref_codes,
                        ref_edge_gate,
                        frac=edge_frac,
                        pretrigger_percent=p.pretrigger_percent,
                        mode=ref_edge_mode,
                        min_run=ref_edge_min_run,
                    )
        return rx_codes, ref_codes, idx_ref, rx_baseline_codes
    
    # ---------- pilot collection ----------
    while len(pilot_data) < target_pilot_count and attempts < max_attempts:
        fr = scope.capture_burst(
            return_mode="raw",
            copy=False,
            apply_vert_cal=False,   # raw anyway
            arm_settle_s=arm_settle_s,
            retries=retries,
        )
        attempts += 1

        if skip_untriggered and not fr.triggered:
            skipped_untriggered += 1
            next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)
            continue

        if skip_overflow and should_skip_for_overflow(
            rx_overflow=fr.rx_overflow,
            ref_overflow=fr.ref_overflow,
            need_ref_processing=need_pilot_ref_processing,
        ):
            skipped_overflow += 1
            next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)
            continue

        if t_s0 is None:
            t_s0 = np.asarray(fr.t_s, dtype=np.float64).copy()
    
        rx_codes, ref_codes, idx_ref, rx_baseline_codes = preprocess_frame(
            fr,
            need_ref_processing=need_pilot_ref_processing,
            need_ref_edge=need_pilot_ref_edge,
            t_s=t_s0,
        )

        if need_pilot_ref_edge and idx_ref is None:
            if skip_missing_ref_edge:
                skipped_missing_ref_edge += 1
                next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)
                continue

        if idx_ref is not None:
            edge_idxs.append(idx_ref)

        pilot_data.append((rx_codes, ref_codes, idx_ref, rx_baseline_codes))

        next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)

    if not pilot_data:
        raise RuntimeError(
            "No valid pilot frames collected. "
            f"skipped_overflow={skipped_overflow}, "
            f"skipped_untriggered={skipped_untriggered}, "
            f"skipped_missing_ref_edge={skipped_missing_ref_edge}"
        )
    assert t_s0 is not None


    edge_arr = np.asarray(edge_idxs, dtype=np.float64)
    has_edge = edge_arr.size > 0

    if has_edge:
        ref_edge_std = float(np.std(edge_arr, ddof=0))
        target_idx = int(round(float(np.mean(edge_arr))))
    else:
        ref_edge_std = float("nan")
        target_idx = None

    if rebasing == "always":
        if target_idx is None:
            raise RuntimeError("rebasing='always', but no reference edge was detected")
        use_rebasing = True
    elif rebasing == "auto":
        use_rebasing = (target_idx is not None) and (ref_edge_std > jitter_threshold_samples)
    else:
        use_rebasing = False

    need_stream_ref_edge = use_rebasing
    need_stream_ref_processing = return_ref or need_stream_ref_edge

    if use_rebasing:
        pilot_data_rebased : list[Tuple[np.ndarray, Optional[np.ndarray], Optional[int], float]] = []

        for rx_codes, ref_codes, idx_ref, rx_baseline_codes in pilot_data:
            if idx_ref is None:
                skipped_missing_ref_edge += 1
                continue
            pilot_data_rebased.append((rx_codes, ref_codes, idx_ref, rx_baseline_codes))

        pilot_data = pilot_data_rebased

        if not pilot_data:
            raise RuntimeError(
                "All pilot frames became invalid for rebasing after filtering missing reference edges"
            )
    
    # ---------- accumulation ----------
    N = len(t_s0)
    sum_rx = np.zeros(N, dtype=np.float64)
    w_rx = np.zeros(N, dtype=np.float64)

    sum_ref = np.zeros(N, dtype=np.float64) if return_ref else None
    w_ref = np.zeros(N, dtype=np.float64) if return_ref else None

    def accumulate_one(rx_codes: np.ndarray, ref_codes: Optional[np.ndarray], idx_ref: int | None) -> None:
        if use_rebasing:
            if target_idx is None or idx_ref is None:
                raise RuntimeError("Internal error: rebasing requires valid idx_ref and target_idx")
            shift = target_idx - idx_ref
            accumulate_shifted(sum_rx, rx_codes, w_rx, shift)

            if return_ref:
                assert ref_codes is not None
                assert sum_ref is not None and w_ref is not None
                accumulate_shifted(sum_ref, ref_codes, w_ref, shift)
        else:
            sum_rx[:] += rx_codes
            w_rx[:] += 1.0

            if return_ref:
                assert ref_codes is not None
                assert sum_ref is not None and w_ref is not None
                sum_ref[:] += ref_codes
                w_ref[:] += 1.0

    for rx_codes, ref_codes, idx_ref, rx_baseline_codes in pilot_data:
        accumulate_one(rx_codes, ref_codes, idx_ref)
        if return_baseline_stats:
            rx_baseline_codes_all.append(rx_baseline_codes)
        accepted_count += 1

    # ---------- remaining frames ----------
    while accepted_count < K and attempts < max_attempts:
        fr = scope.capture_burst(
            return_mode="raw",
            copy=False,
            apply_vert_cal=False,
            arm_settle_s=arm_settle_s,
            retries=retries,
        )
        attempts += 1

        if skip_untriggered and not fr.triggered:
            skipped_untriggered += 1
            next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)
            continue
        
        if skip_overflow and should_skip_for_overflow(
            rx_overflow=fr.rx_overflow,
            ref_overflow=fr.ref_overflow,
            need_ref_processing=need_stream_ref_processing,
        ):
            skipped_overflow += 1
            next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)
            continue

        rx_codes, ref_codes, idx_ref, rx_baseline_codes = preprocess_frame(
            fr,
            need_ref_processing=need_stream_ref_processing,
            need_ref_edge=need_stream_ref_edge,
            t_s=t_s0,
        )

        if need_stream_ref_edge and idx_ref is None:
            skipped_missing_ref_edge += 1
            next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)
            continue

        accumulate_one(rx_codes, ref_codes, idx_ref)
        if return_baseline_stats:
            rx_baseline_codes_all.append(rx_baseline_codes)
        accepted_count += 1

        next_deadline = _sleep_to_next_deadline(next_deadline, repeat_s)

    if accepted_count < K:
        raise RuntimeError(
            f"Only {accepted_count} valid frames were accumulated out of requested {K}, "
            f"attempts={attempts}, skipped_overflow={skipped_overflow}, "
            f"skipped_untriggered={skipped_untriggered}, "
            f"skipped_missing_ref_edge={skipped_missing_ref_edge}"
        )
    
    if return_ref:
        assert w_ref is not None
        valid_mask = (w_rx > 0) & (w_ref > 0)
    else:
        valid_mask = (w_rx > 0)

    if not np.any(valid_mask):
        raise RuntimeError("No valid overlap after rebasing accumulation")
    
    lo = int(np.argmax(valid_mask))
    hi = int(len(valid_mask) - np.argmax(valid_mask[::-1]))

    mean_rx_codes = sum_rx[lo:hi] / w_rx[lo:hi]
    mean_ref_codes = None
    if return_ref:
        assert sum_ref is not None and w_ref is not None
        mean_ref_codes = sum_ref[lo:hi] / w_ref[lo:hi]
    
    t_s = t_s0[lo:hi]

    if use_rebasing:
        # t_rebased_s, idx_final = rebase_time_to_reference(t_s, mean_ref_codes, frac=edge_frac)
        assert target_idx is not None
        idx_final = int(target_idx - lo)
        if not (0 <= idx_final < len(t_s)):
            raise RuntimeError(f"target_idx={target_idx} is outside cropped interval [{lo}, {hi})")
        t_rebased_s = np.asarray(t_s, dtype=np.float64) - t_s[idx_final]
    else:
        t_rebased_s = np.asarray(t_s, dtype=np.float64).copy()
        idx_final = None

    rx_codes_out  = mean_rx_codes
    ref_codes_out = mean_ref_codes if return_ref else None
    rx_volts_out  = None
    ref_volts_out = None

    if output_mode in {"nominal_volts", "calibrated_volts"}:
        rx_volts_out = codes_to_volts(mean_rx_codes, scale_v_per_code=rx_scale_v_per_code)
        if return_ref and mean_ref_codes is not None:
            ref_volts_out = codes_to_volts(mean_ref_codes, scale_v_per_code=ref_scale_v_per_code)

    rx_baseline_mean_codes = None
    rx_baseline_std_codes = None
    rx_baseline_mean_v = None
    rx_baseline_std_v  = None

    if return_baseline_stats:
        rx_baseline_codes_arr  = np.asarray(rx_baseline_codes_all, dtype=np.float64)
        rx_baseline_mean_codes = float(np.mean(rx_baseline_codes_arr))
        rx_baseline_std_codes  = float(np.std(rx_baseline_codes_arr, ddof=0))
        if output_mode in {"nominal_volts", "calibrated_volts"}:
            rx_baseline_mean_v = float(rx_baseline_mean_codes * rx_scale_v_per_code)
            rx_baseline_std_v = float(rx_baseline_std_codes * rx_scale_v_per_code)

    return AveragedBurstResult(
        count=K,
        accepted_count=accepted_count,
        skipped_overflow=skipped_overflow,
        skipped_untriggered=skipped_untriggered,
        skipped_missing_ref_edge=skipped_missing_ref_edge,
        used_rebasing=use_rebasing,
        rebasing_mode_requested=rebasing,
        ref_edge_std_samples=ref_edge_std,
        fs_hz=scope.fs_hz,
        dt_s=scope.dt_s,
        t_s=t_s,
        t_rebased_s=t_rebased_s,
        rx_codes=rx_codes_out,
        ref_codes=ref_codes_out,
        rx_volts=rx_volts_out,
        ref_volts=ref_volts_out,
        ref_edge_index=idx_final,
        rx_baseline_mean_codes=rx_baseline_mean_codes,
        rx_baseline_std_codes=rx_baseline_std_codes,
        rx_baseline_mean_v=rx_baseline_mean_v,
        rx_baseline_std_v=rx_baseline_std_v,
    )


def capture_scan_point_fast(
    scope: HantekHardDll,
    *,
    K: int,
    repeat_s: float,
    edge_frac: float = 0.5,
    rebasing: Literal["always", "never"] = "never",
    pilot_frames: int = 32,
    jitter_threshold_samples: float = 1.0,
    arm_settle_s: float = 0.001,
    retries: int = 2,
    baseline_mode: Literal["none", "rx_pretrigger", "both_pretrigger"] = "rx_pretrigger",
    skip_overflow: bool = True,
    skip_untriggered: bool = True,
    skip_missing_ref_edge: bool = True,
    max_attempt_factor: int = 3,
    ref_edge_gate: Optional[TimeGate] = None,
    ref_edge_min_run: int = 2,
    copy: bool = True,
) -> ScanPointPayload:
    """
    Высокоуровневая обёртка для одной точки сетки
    Всегда возвращает centered codes и не тащит наружу весь AveragedBurstResult
    """
    res = capture_average_burst_fast(
        scope,
        K=K,
        repeat_s=repeat_s,
        edge_frac=edge_frac,
        rebasing=rebasing,
        pilot_frames=pilot_frames,
        jitter_threshold_samples=jitter_threshold_samples,
        arm_settle_s=arm_settle_s,
        retries=retries,
        baseline_mode=baseline_mode,
        skip_overflow=skip_overflow,
        skip_untriggered=skip_untriggered,
        skip_missing_ref_edge=skip_missing_ref_edge,
        output_mode="codes",
        max_attempt_factor=max_attempt_factor,
        return_ref=False,
        ref_edge_gate=ref_edge_gate,
        ref_edge_mode="rise",
        ref_edge_min_run=ref_edge_min_run,
        return_baseline_stats=False
    )

    if res.rx_codes is None:
        raise RuntimeError("capture_average_burst_fast returned no rx_codes in output_mode='codes'")

    rx_codes = np.asarray(res.rx_codes, dtype=np.float32)
    t_s = np.asarray(res.t_rebased_s, dtype=np.float64)

    if copy:
        rx_codes = rx_codes.copy()
        t_s = t_s.copy()

    return ScanPointPayload(
        rx_codes=rx_codes,
        t_s=t_s,
        fs_hz=float(res.fs_hz),
        dt_s=float(res.dt_s),
    )