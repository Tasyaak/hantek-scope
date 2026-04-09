import time, numpy as np, math, statistics
from hantek_scope import (
    HantekHardDll,
    DDSParams,
    DDSWaveType,
    DDSMode,
    StartControl,
    make_scan_params_from_db,
    get_default_dll_dir,
)

DLL_DIR = get_default_dll_dir()

PROFILE = "probe_ch1"
REF_VDIV = "1V"
RX_VDIV_LIST = ["100mV", "200mV", "500mV", "1V"]

K_IDLE = 8
K_ACTIVE = 4

SETTLE_AFTER_CONFIG_S = 0.15
SETTLE_AFTER_DDS_S = 0.10
DISCARD_FRAMES = 5

CHECK_POINTS = {
    "100mV": [
        {"freq_hz": 10_000.0, "amplitude_mv": 100, "offset_mv": 0, "expected_vpp_v": 0.200},
    ],
    "200mV": [
        {"freq_hz": 10_000.0, "amplitude_mv": 200, "offset_mv": 0, "expected_vpp_v": 0.400},
    ],
    "500mV": [
        {"freq_hz": 10_000.0, "amplitude_mv": 500, "offset_mv": 0, "expected_vpp_v": 1.000},
    ],
    "1V": [
        {"freq_hz": 10_000.0, "amplitude_mv": 1000, "offset_mv": 0, "expected_vpp_v": 2.000},
    ],
}


def make_zero_output_mode() -> DDSParams:
    return DDSParams(
        frequency_hz=1000.0,
        amplitude_mv=0,
        offset_mv=0,
        wave_type=DDSWaveType.SINE,
        phase_frac=0.0,
        burst_cycles=1,
        mode=DDSMode.CONTINUOUS,
        duty=0.5,
    )


def make_cont_sine(freq_hz: float, amp_mv: int, offset_mv: int) -> DDSParams:
    return DDSParams(
        frequency_hz=freq_hz,
        amplitude_mv=amp_mv,
        offset_mv=offset_mv,
        wave_type=DDSWaveType.SINE,
        phase_frac=0.0,
        burst_cycles=1,
        mode=DDSMode.CONTINUOUS,
        duty=0.5,
    )


def settle_scope(scope: HantekHardDll) -> None:
    time.sleep(SETTLE_AFTER_CONFIG_S)
    for _ in range(DISCARD_FRAMES):
        _ = scope.capture(return_mode="raw", copy=True, apply_vert_cal=False)


def settle_dds(scope: HantekHardDll) -> None:
    time.sleep(SETTLE_AFTER_DDS_S)
    for _ in range(DISCARD_FRAMES):
        _ = scope.capture(return_mode="raw", copy=True, apply_vert_cal=False)


def signal_stats(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    mean = float(np.mean(y))
    pp = float(np.max(y) - np.min(y))
    rms_ac = float(np.sqrt(np.mean((y - mean) ** 2)))
    return {"mean": mean, "pp": pp, "rms_ac": rms_ac}


def expected_rms_from_vpp(expected_vpp_v: float) -> float:
    return float(expected_vpp_v / (2.0 * math.sqrt(2.0)))


def idle_check() -> None:
    print("\n=== IDLE CHECK ===")
    print("Generator ON in zero-output mode. Corrected mean should be close to zero.")

    zero_dds = make_zero_output_mode()

    with HantekHardDll(DLL_DIR, channel_mode=4) as scope:
        scope.dds_configure(zero_dds, force=True)
        scope.dds_output(True)
        time.sleep(0.2)

        try:
            for rx_vdiv in RX_VDIV_LIST:
                scan = make_scan_params_from_db(
                    PROFILE,
                    rx_volt_div=rx_vdiv,
                    ref_volt_div=REF_VDIV,
                    time_div="200us",
                    read_len=0x4000,
                    pretrigger_percent=20,
                    start_control=StartControl.AUTO,
                )
                scope.configure(scan)
                settle_scope(scope)

                rx_means = []
                ref_means = []

                for _ in range(K_IDLE):
                    fr = scope.capture(return_mode="volts", copy=True, apply_vert_cal=True)
                    assert fr.rx_volts is not None
                    assert fr.ref_volts is not None
                    rx_means.append(float(np.mean(fr.rx_volts)))
                    ref_means.append(float(np.mean(fr.ref_volts)))

                print(f"\nrx_vdiv={rx_vdiv}, ref_vdiv={REF_VDIV}")
                print(f"  corrected mean RX  = {statistics.mean(rx_means):+.9f} V")
                print(f"  corrected mean REF = {statistics.mean(ref_means):+.9f} V")
        finally:
            scope.dds_stop()


def active_check() -> None:
    print("\n=== ACTIVE CHECK ===")
    print("Generator ON with known sine points. Measured Vpp and AC RMS should match expected.")

    with HantekHardDll(DLL_DIR, channel_mode=4) as scope:
        try:
            for rx_vdiv in RX_VDIV_LIST:
                scan = make_scan_params_from_db(
                    PROFILE,
                    rx_volt_div=rx_vdiv,
                    ref_volt_div=REF_VDIV,
                    time_div="200us",
                    read_len=0x4000,
                    pretrigger_percent=20,
                    start_control=StartControl.AUTO,
                )
                scope.configure(scan)
                settle_scope(scope)

                for pt in CHECK_POINTS[rx_vdiv]:
                    dds = make_cont_sine(
                        freq_hz=float(pt["freq_hz"]),
                        amp_mv=int(pt["amplitude_mv"]),
                        offset_mv=int(pt["offset_mv"]),
                    )
                    scope.dds_configure(dds, force=True)
                    scope.dds_output(True)
                    settle_dds(scope)

                    exp_vpp = float(pt["expected_vpp_v"])
                    exp_rms = expected_rms_from_vpp(exp_vpp)

                    means = []
                    pps = []
                    rmsacs = []

                    for _ in range(K_ACTIVE):
                        fr = scope.capture(return_mode="volts", copy=True, apply_vert_cal=True)
                        assert fr.rx_volts is not None
                        st = signal_stats(fr.rx_volts)
                        means.append(st["mean"])
                        pps.append(st["pp"])
                        rmsacs.append(st["rms_ac"])

                    mean_mean = statistics.mean(means)
                    mean_pp = statistics.mean(pps)
                    mean_rms = statistics.mean(rmsacs)

                    err_vpp_pct = 100.0 * (mean_pp - exp_vpp) / exp_vpp if exp_vpp != 0 else float("nan")
                    err_rms_pct = 100.0 * (mean_rms - exp_rms) / exp_rms if exp_rms != 0 else float("nan")

                    print(f"\nrx_vdiv={rx_vdiv}, freq={pt['freq_hz']:.0f}Hz")
                    print(f"  corrected mean RX   = {mean_mean:+.9f} V")
                    print(f"  measured  Vpp RX    = {mean_pp:.9f} V")
                    print(f"  expected  Vpp RX    = {exp_vpp:.9f} V")
                    print(f"  Vpp error           = {err_vpp_pct:+.3f} %")
                    print(f"  measured  AC RMS RX = {mean_rms:.9f} V")
                    print(f"  expected  AC RMS RX = {exp_rms:.9f} V")
                    print(f"  RMS error           = {err_rms_pct:+.3f} %")
        finally:
            scope.dds_stop()


def main() -> None:
    idle_check()
    active_check()


if __name__ == "__main__":
    main()