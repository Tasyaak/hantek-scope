import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .enums import (
    Channel, Coupling, TriggerMode, TriggerSlope,
    TriggerCoupling, YTFormat, StartControl, DDSWaveType, DDSMode
)


@dataclass(frozen=True)
class VertCal:
    zero_code_abs : float
    scale_v_per_code : float

    def __post_init__(self) -> None:
        if not np.isfinite(self.zero_code_abs):
            raise ValueError(f"zero_code_abs must be finite, got {self.zero_code_abs}")
        if not np.isfinite(self.scale_v_per_code) or self.scale_v_per_code <= 0.0:
            raise ValueError(
                f"scale_v_per_code must be finite and > 0, got {self.scale_v_per_code}"
            )
        
    
@dataclass(frozen=True)
class ScanParams:
    # waveform window
    time_div : int | str = "10ms"
    read_len : int = 0x1000
    pretrigger_percent : int = 10
    yt_format : YTFormat = YTFormat.NORMAL

    # receiver channel CH1
    rx_channel : Channel = Channel.CH1
    rx_volt_div : int | str = "1V"
    rx_coupling : Coupling = Coupling.DC
    rx_bw_limit : bool = False
    rx_lever_pos : int = 128
    rx_cal : Optional[VertCal] = None

    # optional reference channel from GEN OUT -> CH2
    ref_channel : Optional[Channel] = Channel.CH2
    ref_volt_div : int | str = "1V"
    ref_coupling : Coupling = Coupling.DC
    ref_bw_limit : bool = False
    ref_lever_pos : int = 128
    ref_cal : Optional[VertCal] = None

    # trigger
    trig_source : Channel = Channel.CH2
    trig_mode : TriggerMode = TriggerMode.EDGE
    trig_slope : TriggerSlope = TriggerSlope.RISE
    trig_level_pos : int = 128
    trigger_coupling : TriggerCoupling = TriggerCoupling.DC
    trigger_sensitivity : int = 4

    # acquisition control
    start_control : StartControl = StartControl.NONE
    timeout_s : float = 2.0
    poll_s : float = 0.001

    def __post_init__(self) -> None:
        if not (0 <= self.pretrigger_percent <= 100):
            raise ValueError("pretrigger_percent must be in [0; 100]")

        if self.read_len <= 0:
            raise ValueError("read_len must be > 0")
        if self.read_len % 512 != 0:
            raise ValueError("read_len must be a multiple of 512")
        if self.read_len > 16 * 1024 * 1024:
            raise ValueError("read_len must be <= 16M samples")

        for name, value in (
            ("rx_lever_pos", self.rx_lever_pos),
            ("ref_lever_pos", self.ref_lever_pos),
            ("trig_level_pos", self.trig_level_pos),
            ("trigger_sensitivity", self.trigger_sensitivity),
        ):
            if not (0 <= value <= 255):
                raise ValueError(f"{name} must be in [0; 255], got {value}")

        enabled = {self.rx_channel}
        if self.ref_channel is not None:
            enabled.add(self.ref_channel)

        if self.trig_source not in enabled:
            raise ValueError("trig_source must be one of enabled channels")

        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.poll_s <= 0:
            raise ValueError("poll_s must be > 0")
        
        if self.rx_cal is not None and not isinstance(self.rx_cal, VertCal):
            raise TypeError("rx_cal must be VertCal or None")
        if self.ref_cal is not None and not isinstance(self.ref_cal, VertCal):
            raise TypeError("ref_cal must be VertCal or None")


@dataclass(frozen=True)
class DDSParams:
    frequency_hz : float
    amplitude_mv : int
    offset_mv : int = 0
    wave_type : DDSWaveType = DDSWaveType.SINE
    phase_frac : float = 0.0
    duty : float = 0.25
    burst_cycles : int = 1
    mode: DDSMode = DDSMode.CONTINUOUS

    def __post_init__(self) -> None:
        if not (0.0 < self.frequency_hz <= 25e6):
            raise ValueError("frequency_hz must be in (0.0; 25000000] Hz")
        if not (0.0 <= self.amplitude_mv <= 3500.0):
            raise ValueError("amplitude_mv must be in [0.0; 3500.0] mV")
        if not (-3500 <= self.offset_mv <= 3500):
            raise ValueError("offset_mv must be in [-3500; 3500] mV")
        if self.amplitude_mv + abs(self.offset_mv) > 3500:
            raise ValueError("amplitude_mv + abs(offset_mv) must be <= 3500 mV")
        if not (0.0 <= self.phase_frac <= 1.0):
            raise ValueError("phase_frac must be in [0.0; 1.0]")
        if not (0.0 <= self.duty <= 1.0):
            raise ValueError("duty must be in [0.0; 1.0]")
        if self.burst_cycles < 1:
            raise ValueError("burst_cycles must be >= 1")


@dataclass(frozen=True)
class AScanFrame:
    """
    t и raw могут быть view на внутренний буфер (если copy=False), при следующем capture() данные будут перезаписаны
    t — кэшируемый массив времени (не меняется при фиксированных параметрах)
    """
    point : Optional[Tuple[float, float]]
    fs_hz : float
    dt_s : float
    t_s : np.ndarray
    triggered : bool
    rx_volts : Optional[np.ndarray] = None
    rx_raw_u16 : Optional[np.ndarray] = None
    ref_volts : Optional[np.ndarray] = None
    ref_raw_u16 : Optional[np.ndarray] = None
    rx_overflow : bool = False
    ref_overflow : bool = False


@dataclass
class AveragedBurstResult:
    count : int
    accepted_count : int
    skipped_overflow : int
    skipped_untriggered : int
    skipped_missing_ref_edge : int

    used_rebasing : bool
    rebasing_mode_requested : str
    ref_edge_std_samples : float
    
    fs_hz : float
    dt_s : float

    t_s : np.ndarray
    t_rebased_s : np.ndarray

    rx_codes : Optional[np.ndarray] = None
    ref_codes : Optional[np.ndarray] = None

    rx_volts : Optional[np.ndarray] = None
    ref_volts : Optional[np.ndarray] = None
    ref_edge_index : Optional[int] = None

    rx_baseline_mean_codes : Optional[float] = None
    rx_baseline_std_codes : Optional[float] = None
    rx_baseline_mean_v : Optional[float] = None
    rx_baseline_std_v : Optional[float] = None


@dataclass(frozen=True)
class ScanPointPayload:
    """
    Компактный результат для одной точки сетки

    rx_codes : усреднённый A-scan в centered codes
    t_s      : временная ось для этого A-scan
    fs_hz    : частота дискретизации
    dt_s     : шаг по времени
    """
    rx_codes : np.ndarray
    t_s : np.ndarray
    fs_hz : float
    dt_s : float

    def __post_init__(self) -> None:
        rx = np.asarray(self.rx_codes)
        t = np.asarray(self.t_s)

        if rx.ndim != 1:
            raise ValueError(f"rx_codes must be 1D, got shape={rx.shape}")
        if t.ndim != 1:
            raise ValueError(f"t_s must be 1D, got shape={t.shape}")
        if rx.shape[0] != t.shape[0]:
            raise ValueError(
                f"rx_codes and t_s must have the same length, got "
                f"{rx.shape[0]} and {t.shape[0]}"
            )
        if not np.isfinite(self.fs_hz) or self.fs_hz <= 0.0:
            raise ValueError(f"fs_hz must be finite and > 0, got {self.fs_hz}")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be finite and > 0, got {self.dt_s}")


class HantekError(RuntimeError):
    pass


def _check_ok(name : str, ok : int) -> None:
    if int(ok) == 0:
        raise HantekError(f"{name} failed (returned 0)")