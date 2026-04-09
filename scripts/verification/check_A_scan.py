import time, numpy as np, matplotlib.pyplot as plt
from hantek_scope import (
    HantekHardDll,
    ScanParams,
    DDSParams,
    DDSWaveType,
    DDSMode,
    StartControl,
    make_scan_params_nominal,
    get_default_dll_dir,
    capture_average_burst_fast,
    TimeGate,
    peak_abs_in_gate,
    rms_in_gate,
    rms_ac_in_gate,
    argmax_abs_index_in_gate,
    extract_in_gate,
)


DLL_DIR = get_default_dll_dir()

# -------------------------------------------------------
# Настройки эксперимента
# -------------------------------------------------------

K_AVG = 15000
REPEAT_S = 0.013
ARM_SETTLE_S = 0.001
RETRIES = 2

# Режим только для оценки CH1
OUTPUT_MODE = "codes"
REBASE_MODE = "never"

# Рабочие окна
NOISE_GATE = TimeGate(t1_s=-40e-6, t2_s=-2e-6, name="noise_gate")
PROMPT_GATE = TimeGate(t1_s=-2e-6, t2_s=15e-6, name="prompt_gate")
RX_GATE = TimeGate(t1_s=15e-6, t2_s=300e-6, name="rx_gate")

SCAN = make_scan_params_nominal(
    time_div="5us",
    read_len=0x8000,
    pretrigger_percent=10,
    rx_volt_div="20mV",
    rx_lever_pos=128,
    ref_volt_div="1V",
    ref_lever_pos=128,
    start_control=StartControl.NONE
)

DDS = DDSParams(
    frequency_hz=150_000.0,
    amplitude_mv=3500,
    offset_mv=0,
    wave_type=DDSWaveType.SQUARE,
    duty=0.4,
    burst_cycles=1,
    mode=DDSMode.BURST,
)


def require_array(name: str, x: np.ndarray | None) -> np.ndarray:
    if x is None:
        raise RuntimeError(f"{name} is None")
    return np.asarray(x, dtype=np.float64)


def plot_gate(ax, gate: TimeGate, *, color: str, alpha: float = 0.10) -> None:
    ax.axvspan(gate.t1_s * 1e6, gate.t2_s * 1e6, color=color, alpha=alpha)
    if gate.name:
        xc = 0.5 * (gate.t1_s + gate.t2_s) * 1e6
        ax.text(
            xc,
            0.98,
            gate.name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )


def print_capture_window(scope: HantekHardDll, params: ScanParams) -> None:
    t_total = params.read_len / scope.fs_hz
    t_pre = params.pretrigger_percent / 100.0 * t_total
    t_post = t_total - t_pre

    print("=== CAPTURE WINDOW ===")
    print(f"K_AVG={K_AVG}")
    print(f"rebasing={REBASE_MODE}")
    print(f"time_div={params.time_div}")
    print(f"rx_volt_div={params.rx_volt_div}")
    print(f"frequency_hz={DDS.frequency_hz} Hz")
    print(f"amplitude_mv={DDS.amplitude_mv} mV")
    print(f"duty={DDS.duty}\n")

    print(f"fs_hz={scope.fs_hz:.6e} Hz")
    print(f"dt_s={scope.dt_s:.6e} s")
    print(f"read_len={params.read_len}")
    print(f"T_total={t_total * 1e6:.3f} us")
    print(f"T_pre={t_pre * 1e6:.3f} us")
    print(f"T_post={t_post * 1e6:.3f} us\n")


def summarize_signal(t_s: np.ndarray, rx: np.ndarray, res) -> dict[str, float]:
    peak_full = float(np.max(np.abs(rx)))
    mean_full = float(np.mean(rx))
    std_full = float(np.std(rx, ddof=0))
    rms_full = float(np.sqrt(np.mean(rx ** 2)))

    noise_mean = float("nan")
    noise_std = float("nan")
    noise_rms = float("nan")
    noise_rms_ac = float("nan")

    rx_peak = float("nan")
    rx_rms = float("nan")
    rx_rms_ac = float("nan")
    prompt_peak = float("nan")
    contrast = float("nan")
    prompt_to_rx_ratio = float("nan")
    t_peak_rx_us = float("nan")

    # noise
    _, noise_seg = extract_in_gate(t_s, rx, NOISE_GATE)
    if noise_seg.size:
        noise_mean = float(np.mean(noise_seg))
        noise_std = float(np.std(noise_seg, ddof=0))
        noise_rms = rms_in_gate(t_s, rx, NOISE_GATE)
        noise_rms_ac = rms_ac_in_gate(t_s, rx, NOISE_GATE)

    # prompt
    prompt_peak = peak_abs_in_gate(t_s, rx, PROMPT_GATE)

    # useful RX window
    rx_peak = peak_abs_in_gate(t_s, rx, RX_GATE)
    rx_rms = rms_in_gate(t_s, rx, RX_GATE)
    rx_rms_ac = rms_ac_in_gate(t_s, rx, RX_GATE)

    idx_peak = argmax_abs_index_in_gate(t_s, rx, RX_GATE)
    if idx_peak is not None:
        t_peak_rx_us = float(t_s[idx_peak] * 1e6)

    if np.isfinite(rx_rms_ac) and np.isfinite(noise_rms_ac):
        contrast = rx_rms_ac / max(noise_rms_ac, 1e-15)

    if np.isfinite(prompt_peak) and np.isfinite(rx_peak):
        prompt_to_rx_ratio = prompt_peak / max(rx_peak, 1e-15)

    out = {
        "peak_full": peak_full,
        "mean_full": mean_full,
        "std_full": std_full,
        "rms_full": rms_full,

        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "noise_rms": noise_rms,
        "noise_rms_ac": noise_rms_ac,

        "prompt_peak": prompt_peak,

        "rx_peak": rx_peak,
        "rx_rms": rx_rms,
        "rx_rms_ac": rx_rms_ac,
        "contrast": contrast,
        "t_peak_rx_us": t_peak_rx_us,
        "prompt_to_rx_ratio": prompt_to_rx_ratio,

        "baseline_mean_codes": (
            float(res.rx_baseline_mean_codes)
            if res.rx_baseline_mean_codes is not None else float("nan")
        ),
        "baseline_std_codes": (
            float(res.rx_baseline_std_codes)
            if res.rx_baseline_std_codes is not None else float("nan")
        ),
    }
    return out


def print_summary(res, stats: dict[str, float]) -> None:
    print("=== AVERAGED CH1 SUMMARY ===")
    print(f"used_rebasing              : {res.used_rebasing}")
    print(f"accepted_count             : {res.accepted_count} / {res.count}")
    print(f"skipped_untriggered        : {res.skipped_untriggered}")
    print(f"skipped_overflow           : {res.skipped_overflow}")
    print(f"skipped_missing_ref_edge   : {res.skipped_missing_ref_edge}")
    print()

    print(f"|peak| full trace          : {stats['peak_full']:.6f} codes")
    print(f"mean full trace            : {stats['mean_full']:.6f} codes")
    print(f"std  full trace            : {stats['std_full']:.6f} codes")
    print(f"RMS  full trace            : {stats['rms_full']:.6f} codes")
    print()

    print(f"noise mean                 : {stats['noise_mean']:.6f} codes")
    print(f"noise std                  : {stats['noise_std']:.6f} codes")
    print(f"noise RMS                  : {stats['noise_rms']:.6f} codes")
    print(f"noise RMS_ac               : {stats['noise_rms_ac']:.6f} codes")
    print()

    print(f"prompt |peak|              : {stats['prompt_peak']:.6f} codes")
    print(f"rx |peak| in RX_GATE       : {stats['rx_peak']:.6f} codes")
    print(f"rx RMS in RX_GATE          : {stats['rx_rms']:.6f} codes")
    print(f"rx RMS_ac in RX_GATE       : {stats['rx_rms_ac']:.6f} codes")
    print(f"contrast = RX_RMS_ac / NOISE_RMS_ac : {stats['contrast']:.6f}")
    print(f"t_peak in RX_GATE          : {stats['t_peak_rx_us']:.3f} us")
    print(f"prompt_peak / rx_peak      : {stats['prompt_to_rx_ratio']:.6f}")
    print()

    print(f"baseline mean              : {stats['baseline_mean_codes']:.6f} codes")
    print(f"baseline std               : {stats['baseline_std_codes']:.6f} codes")


def make_plot(t_s: np.ndarray, rx: np.ndarray, stats: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(t_s * 1e6, rx, label='avg CH1, rebasing="never"')
    ax.axhline(0.0, linewidth=1.0)

    plot_gate(ax, NOISE_GATE, color="tab:purple", alpha=0.08)
    plot_gate(ax, PROMPT_GATE, color="tab:red", alpha=0.08)
    plot_gate(ax, RX_GATE, color="tab:orange", alpha=0.10)

    if np.isfinite(stats["t_peak_rx_us"]):
        ax.axvline(
            stats["t_peak_rx_us"],
            linestyle="--",
            linewidth=1.0,
            label=f'peak in RX_GATE: {stats["t_peak_rx_us"]:.2f} us'
        )

    text = (
        f'|peak|_RX = {stats["rx_peak"]:.5f} codes\n'
        f'RMS_ac_RX = {stats["rx_rms_ac"]:.5f} codes\n'
        f'RMS_ac_noise = {stats["noise_rms_ac"]:.5f} codes\n'
        f'contrast = {stats["contrast"]:.3f}\n'
        f'baseline std = {stats["baseline_std_codes"]:.5f} codes'
    )

    ax.text(
        0.985, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    ax.set_title('Averaged CH1 receiver signal, rebasing="never"')
    ax.set_xlabel("time, us")
    ax.set_ylabel("codes")
    ax.legend()
    fig.tight_layout()
    plt.show()


def main() -> None:
    with HantekHardDll(DLL_DIR) as scope:
        scope.configure(SCAN)
        print_capture_window(scope, SCAN)

        scope.dds_configure(DDS, force=True)
        scope.dds_output(True)
        time.sleep(0.2)

        try:
            res = capture_average_burst_fast(
                scope,
                K=K_AVG,
                repeat_s=REPEAT_S,
                rebasing=REBASE_MODE,
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
                return_ref=False,
                ref_edge_min_run=2,
                return_baseline_stats=True,
            )
        finally:
            scope.dds_stop()

    rx = require_array("res.rx_codes", res.rx_codes)
    t_s = np.asarray(res.t_s, dtype=np.float64)

    stats = summarize_signal(t_s, rx, res)
    print_summary(res, stats)
    make_plot(t_s, rx, stats)


if __name__ == "__main__":
    main()