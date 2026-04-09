import numpy as np
from typing import Tuple
from .models import VertCal
from .enums import Channel, Coupling, TriggerMode, TriggerSlope, TriggerCoupling, StartControl
from .models import ScanParams


def raw_to_centered_codes(raw_u16 : np.ndarray, *, zero_code_abs : float) -> np.ndarray:
    raw = raw_u16.astype(np.float64)
    return raw - float(zero_code_abs)

def codes_to_volts(centered_codes : np.ndarray, *, scale_v_per_code : float) -> np.ndarray:
    if scale_v_per_code <= 0.0 or not np.isfinite(scale_v_per_code):
        raise RuntimeError("scale_v_per_code is not initialized correctly")
    return centered_codes * float(scale_v_per_code)


VERT_CAL_DB = {
    "probe_ch1": {
        "rx": {
            "100mV": VertCal(zero_code_abs=129.000000, scale_v_per_code=0.003221133),
            "200mV": VertCal(zero_code_abs=131.687073, scale_v_per_code=0.006619860),
            "500mV": VertCal(zero_code_abs=129.143402, scale_v_per_code=0.016488061),
            "1V": VertCal(zero_code_abs=127.897217, scale_v_per_code=0.033575008),
        },
        "ref": {
            "1V": VertCal(zero_code_abs=128.958145, scale_v_per_code=0.035047284),
        },
    },
    "transmitter_ch1": {
        "rx": {
            
        },
        "ref": {
            
        },
    }
}


def resolve_vert_cal(profile_name : str, *, rx_volt_div : str, ref_volt_div : str) -> Tuple[VertCal, VertCal]:
    prof = VERT_CAL_DB[profile_name]
    if rx_volt_div not in prof["rx"]:
        raise KeyError(
            f"No rx VertCal for profile={profile_name!r}, rx_volt_div={rx_volt_div!r}"
        )
    if ref_volt_div not in prof["ref"]:
        raise KeyError(
            f"No ref VertCal for profile={profile_name!r}, ref_volt_div={ref_volt_div!r}"
        )
    return prof["rx"][rx_volt_div], prof["ref"][ref_volt_div]


def make_scan_params_from_db(
    profile_name : str,
    *,
    rx_volt_div : str,
    ref_volt_div : str = "1V",
    time_div : str = "50us",
    read_len : int = 0x4000,
    pretrigger_percent : int = 10,
    rx_lever_pos : int = 128,
    ref_lever_pos : int = 128,
    start_control : StartControl = StartControl.NONE
) -> ScanParams:
    rx_cal, ref_cal = resolve_vert_cal(
        profile_name,
        rx_volt_div=rx_volt_div,
        ref_volt_div=ref_volt_div,
    )

    return ScanParams(
        time_div=time_div,
        read_len=read_len,
        pretrigger_percent=pretrigger_percent,

        rx_channel=Channel.CH1,
        rx_volt_div=rx_volt_div,
        rx_coupling=Coupling.DC,
        rx_bw_limit=False,
        rx_lever_pos=rx_lever_pos,
        rx_cal=rx_cal,

        ref_channel=Channel.CH2,
        ref_volt_div=ref_volt_div,
        ref_coupling=Coupling.DC,
        ref_bw_limit=False,
        ref_lever_pos=ref_lever_pos,
        ref_cal=ref_cal,

        trig_source=Channel.CH2,
        trig_mode=TriggerMode.EDGE,
        trig_slope=TriggerSlope.RISE,
        trig_level_pos=128,
        trigger_coupling=TriggerCoupling.DC,
        trigger_sensitivity=4,

        start_control=start_control,
        timeout_s=2.0,
        poll_s=0.0005,
    )


def make_scan_params_nominal(
    *,
    rx_volt_div: int | str,
    ref_volt_div: int | str = "1V",
    time_div: int | str = "50us",
    read_len: int = 0x4000,
    pretrigger_percent: int = 10,
    rx_lever_pos: int = 128,
    ref_lever_pos: int = 128,
    start_control: StartControl = StartControl.NONE,
) -> ScanParams:
    return ScanParams(
        time_div=time_div,
        read_len=read_len,
        pretrigger_percent=pretrigger_percent,

        rx_channel=Channel.CH1,
        rx_volt_div=rx_volt_div,
        rx_coupling=Coupling.DC,
        rx_bw_limit=False,
        rx_lever_pos=rx_lever_pos,
        rx_cal=None,

        ref_channel=Channel.CH2,
        ref_volt_div=ref_volt_div,
        ref_coupling=Coupling.DC,
        ref_bw_limit=False,
        ref_lever_pos=ref_lever_pos,
        ref_cal=None,

        trig_source=Channel.CH2,
        trig_mode=TriggerMode.EDGE,
        trig_slope=TriggerSlope.RISE,
        trig_level_pos=128,
        trigger_coupling=TriggerCoupling.DC,
        trigger_sensitivity=4,

        start_control=start_control,
        timeout_s=1.0,
        poll_s=0.0005,
    )