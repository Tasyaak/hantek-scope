import time, math, numpy as np
from hantek_scope import (
    HantekHardDll,
    ScanParams,
    DDSParams,
    DDSWaveType,
    DDSMode,
    Channel,
    Coupling,
    TriggerMode,
    TriggerSlope,
    TriggerCoupling,
    StartControl,
    VertCal,
    get_default_dll_dir,
)

DLL_DIR = get_default_dll_dir()

PROFILE_NAME = "probe_ch1"
REF_VDIV = "1V"
RX_VDIV_LIST = ["50mV", "100mV", "200mV", "500mV"]

ZERO_K = 128
SCALE_K = 32

SETTLE_AFTER_CONFIG_S = 0.15
SETTLE_AFTER_DDS_S = 0.20
DISCARD_FRAMES = 5

# Явно задаём эталонные точки для fit масштаба.
# expected_vpp_v вы заполняете по той модели генератора, которую считаете
# уже эмпирически подтверждённой для своей установки.
SCALE_POINTS = {
    "100mV": [
        {"freq_hz": 10_000.0, "amplitude_mv": 100, "offset_mv": 0, "expected_vpp_v": 0.200},
        {"freq_hz": 10_000.0, "amplitude_mv": 150, "offset_mv": 0, "expected_vpp_v": 0.300},
    ],
    "200mV": [
        {"freq_hz": 10_000.0, "amplitude_mv": 200, "offset_mv": 0, "expected_vpp_v": 0.400},
        {"freq_hz": 10_000.0, "amplitude_mv": 300, "offset_mv": 0, "expected_vpp_v": 0.600},
    ],
    "500mV": [
        {"freq_hz": 10_000.0, "amplitude_mv": 500, "offset_mv": 0, "expected_vpp_v": 1.000},
        {"freq_hz": 10_000.0, "amplitude_mv": 750, "offset_mv": 0, "expected_vpp_v": 1.500},
    ],
    "1V": [
        {"freq_hz": 10_000.0, "amplitude_mv": 500, "offset_mv": 0, "expected_vpp_v": 1.000},
        {"freq_hz": 10_000.0, "amplitude_mv": 1000, "offset_mv": 0, "expected_vpp_v": 2.000},
    ],
}


def make_scan(rx_vdiv : str) -> ScanParams:
    return ScanParams(
        time_div="200us",
        read_len=0x4000,
        pretrigger_percent=20,

        rx_channel=Channel.CH1,
        rx_volt_div=rx_vdiv,
        rx_coupling=Coupling.DC,
        rx_bw_limit=False,
        rx_lever_pos=128,
        rx_cal=None,

        ref_channel=Channel.CH2,
        ref_volt_div=REF_VDIV,
        ref_coupling=Coupling.DC,
        ref_bw_limit=False,
        ref_lever_pos=128,
        ref_cal=None,

        trig_source=Channel.CH2,
        trig_mode=TriggerMode.EDGE,
        trig_slope=TriggerSlope.RISE,
        trig_level_pos=128,
        trigger_coupling=TriggerCoupling.DC,
        trigger_sensitivity=4,

        start_control=StartControl.AUTO,
        timeout_s=2.0,
        poll_s=0.001,
    )


def make_zero_dds() -> DDSParams:
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


def make_sine_dds(freq_hz: float, amplitude_mv: int, offset_mv: int) -> DDSParams:
    return DDSParams(
        frequency_hz=freq_hz,
        amplitude_mv=amplitude_mv,
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


def rms_ac(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    return float(np.sqrt(np.mean(x ** 2)))


def expected_rms_from_vpp(expected_vpp_v: float) -> float:
    # Только для синуса:
    # Vrms(ac) = Vpp / (2 * sqrt(2))
    return float(expected_vpp_v / (2.0 * math.sqrt(2.0)))


def estimate_zero_abs_pair(scope: HantekHardDll, *, K: int) -> tuple[float, float]:
    rx_raw_means = []
    ref_raw_means = []

    for _ in range(K):
        fr = scope.capture(return_mode="raw", copy=True, apply_vert_cal=False)
        assert fr.rx_raw_u16 is not None
        assert fr.ref_raw_u16 is not None

        rx_raw_means.append(float(np.mean(fr.rx_raw_u16)))
        ref_raw_means.append(float(np.mean(fr.ref_raw_u16)))

    rx_raw_means = np.asarray(rx_raw_means, dtype=np.float64)
    ref_raw_means = np.asarray(ref_raw_means, dtype=np.float64)

    rx_zero_abs = float(np.median(rx_raw_means))
    ref_zero_abs = float(np.median(ref_raw_means))

    print("zero abs:")
    print(f"  rx_zero_code_abs  = {rx_zero_abs:.6f}")
    print(f"  ref_zero_code_abs = {ref_zero_abs:.6f}")
    print(f"  rx_std_raw_mean   = {rx_raw_means.std(ddof=0):.6f}")
    print(f"  ref_std_raw_mean  = {ref_raw_means.std(ddof=0):.6f}")

    return rx_zero_abs, ref_zero_abs


def estimate_scale_pair(
    scope: HantekHardDll,
    *,
    rx_zero_code_abs: float,
    ref_zero_code_abs: float,
    rx_vdiv: str,
    K: int,
) -> tuple[float, float]:
    rx_scale_candidates = []
    ref_scale_candidates = []

    for pt in SCALE_POINTS[rx_vdiv]:
        dds = make_sine_dds(
            freq_hz=float(pt["freq_hz"]),
            amplitude_mv=int(pt["amplitude_mv"]),
            offset_mv=int(pt["offset_mv"]),
        )
        scope.dds_configure(dds, force=True)
        settle_dds(scope)

        expected_rms_v = expected_rms_from_vpp(float(pt["expected_vpp_v"]))

        local_rx = []
        local_ref = []

        for _ in range(K):
            fr = scope.capture(return_mode="raw", copy=True, apply_vert_cal=False)
            assert fr.rx_raw_u16 is not None
            assert fr.ref_raw_u16 is not None

            if fr.rx_overflow or fr.ref_overflow:
                continue

            rx_codes = fr.rx_raw_u16.astype(np.float64) - rx_zero_code_abs
            ref_codes = fr.ref_raw_u16.astype(np.float64) - ref_zero_code_abs

            rx_rms_codes = rms_ac(rx_codes)
            ref_rms_codes = rms_ac(ref_codes)

            if rx_rms_codes > 0:
                local_rx.append(expected_rms_v / rx_rms_codes)
            if ref_rms_codes > 0:
                local_ref.append(expected_rms_v / ref_rms_codes)

        if local_rx:
            local_rx = np.asarray(local_rx, dtype=np.float64)
            rx_scale_pt = float(np.median(local_rx))
            rx_scale_candidates.extend(local_rx.tolist())
            print(
                f"  RX point freq={pt['freq_hz']:.0f}Hz, expected_Vpp={pt['expected_vpp_v']:.3f}V:"
                f" scale median = {rx_scale_pt:.9f} V/code"
            )
        else:
            print(
                f"  RX point freq={pt['freq_hz']:.0f}Hz, expected_Vpp={pt['expected_vpp_v']:.3f}V:"
                f" no valid non-overflow frames"
            )

        if local_ref:
            local_ref = np.asarray(local_ref, dtype=np.float64)
            ref_scale_pt = float(np.median(local_ref))
            ref_scale_candidates.extend(local_ref.tolist())
            print(
                f"  REF point freq={pt['freq_hz']:.0f}Hz, expected_Vpp={pt['expected_vpp_v']:.3f}V:"
                f" scale median = {ref_scale_pt:.9f} V/code"
            )
        else:
            print(
                f"  REF point freq={pt['freq_hz']:.0f}Hz, expected_Vpp={pt['expected_vpp_v']:.3f}V:"
                f" no valid non-overflow frames"
            )

    if not rx_scale_candidates:
        raise RuntimeError(f"No valid RX scale candidates for {rx_vdiv}")
    if not ref_scale_candidates:
        raise RuntimeError(f"No valid REF scale candidates for {rx_vdiv}")

    rx_scale = float(np.median(np.asarray(rx_scale_candidates, dtype=np.float64)))
    ref_scale = float(np.median(np.asarray(ref_scale_candidates, dtype=np.float64)))

    return rx_scale, ref_scale


def main() -> None:
    print(f"PROFILE_NAME = {PROFILE_NAME}")
    print("Calibration plan:")
    print("  1) active-zero generator mode -> zero_code_abs")
    print("  2) several sine points without overflow -> scale_v_per_code")
    print("  3) print ready-to-paste VERT_CAL_DB block")

    rx_db = {}
    ref_zero_abs_all = []
    ref_scale_all = []

    with HantekHardDll(DLL_DIR, channel_mode=4) as scope:
        zero_dds = make_zero_dds()
        scope.dds_configure(zero_dds, force=True)
        scope.dds_output(True)
        time.sleep(0.2)

        for rx_vdiv in RX_VDIV_LIST:
            print(f"\n===== RX VDIV = {rx_vdiv} =====")
            scan = make_scan(rx_vdiv)
            scope.configure(scan)
            settle_scope(scope)

            rx_zero_abs, ref_zero_abs = estimate_zero_abs_pair(scope, K=ZERO_K)

            rx_scale, ref_scale = estimate_scale_pair(
                scope,
                rx_zero_code_abs=rx_zero_abs,
                ref_zero_code_abs=ref_zero_abs,
                rx_vdiv=rx_vdiv,
                K=SCALE_K,
            )

            rx_db[rx_vdiv] = VertCal(
                zero_code_abs=rx_zero_abs,
                scale_v_per_code=rx_scale,
            )
            ref_zero_abs_all.append(ref_zero_abs)
            ref_scale_all.append(ref_scale)

            print(f"  FINAL RX zero_code_abs    = {rx_zero_abs:.6f}")
            print(f"  FINAL RX scale_v_per_code = {rx_scale:.9f}")

        scope.dds_stop()

    ref_cal = VertCal(
        zero_code_abs=float(np.median(np.asarray(ref_zero_abs_all, dtype=np.float64))),
        scale_v_per_code=float(np.median(np.asarray(ref_scale_all, dtype=np.float64))),
    )

    print("\n===== READY TO PASTE =====")
    print(f'"{PROFILE_NAME}": {{')
    print('    "rx": {')
    for rx_vdiv in RX_VDIV_LIST:
        c = rx_db[rx_vdiv]
        print(
            f'        "{rx_vdiv}": VertCal(zero_code_abs={c.zero_code_abs:.6f}, '
            f'scale_v_per_code={c.scale_v_per_code:.9f}),'
        )
    print("    },")
    print('    "ref": {')
    print(
        f'        "{REF_VDIV}": VertCal(zero_code_abs={ref_cal.zero_code_abs:.6f}, '
        f'scale_v_per_code={ref_cal.scale_v_per_code:.9f}),'
    )
    print("    },")
    print("}")
    

if __name__ == "__main__":
    main()