import time, ctypes as ct, numpy as np
from pathlib import Path
from typing import Tuple, Optional, Literal
from .sdk_structs import WORD, UINT, RELAYCONTROL, CONTROLDATA
from .sdk_bindings import load_hantek_library, bind_hantek_prototypes
from .models import ScanParams, DDSParams, AScanFrame, HantekError, _check_ok
from .parsing import parse_time_div_idx, parse_volt_div_idx
from .constants import MAX_CH_NUM, MAX_DATA, VOLT_DIV_V
from .enums import Channel, DeviceState, DDSWaveType, DDSMode
from .calibration import raw_to_centered_codes, codes_to_volts
from .acquisition import is_overflow


class HantekHardDll:
    def __init__(self, dll_dir : Path, *, channel_mode : int = 4, device_index : Optional[int] = None, max_devices : int = 32) -> None:
        self.channel_mode = int(channel_mode)
        self.max_devices = int(max_devices)

        # WinDLL/windll на Windows использует stdcall-соглашение вызова
        self.lib, self._dll_cookie = load_hantek_library(dll_dir)
        bind_hantek_prototypes(self.lib)
        self.device_index = device_index if device_index is not None else self._find_device()

        # init один раз
        _check_ok("dsoInitHard", self.lib.dsoInitHard(WORD(self.device_index)))
        _check_ok("dsoHTADCCHModGain", self.lib.dsoHTADCCHModGain(WORD(self.device_index), WORD(self.channel_mode)))

        # Будет заполнено при configure()
        self.params : Optional[ScanParams] = None
        self.fs_hz : float = 0.0
        self.dt_s : float = 0.0
        self._ampcal_key : Optional[Tuple] = None
        self._time_div_idx : Optional[int] = None
        self._rx_volt_div_idx : Optional[int] = None
        self._ref_volt_div_idx : Optional[int] = None
        self._scan_key : Optional[Tuple] = None

        self._rx_nominal_zero_code_abs : float = 0.0
        self._ref_nominal_zero_code_abs : float = 0.0
        self._rx_nominal_scale_v_per_code : float = 0.0
        self._ref_nominal_scale_v_per_code : float = 0.0

        self._rx_zero_code_abs : float = 0.0
        self._ref_zero_code_abs : float = 0.0
        self._rx_scale_v_per_code : float = 0.0
        self._ref_scale_v_per_code : float = 0.0

        # Будет заполнено при dds_configure()
        self._dds_params : Optional[DDSParams] = None
        self._dds_on : bool = False
        self._dds_key : Optional[Tuple] = None
        
        # Структуры управления — живут весь скан
        self._relay = RELAYCONTROL()
        self._ctrl = CONTROLDATA()

        # Буферы данных (4 канала) — выделим при configure()
        self._buf_len : int = 0
        self._ch1_buf : Optional[ct.Array[WORD]] = None
        self._ch2_buf : Optional[ct.Array[WORD]] = None
        self._ch3_buf : Optional[ct.Array[WORD]] = None
        self._ch4_buf : Optional[ct.Array[WORD]] = None

        # Кэш времени (при фиксированных параметрах строится один раз)
        self._t_cache : Optional[np.ndarray] = None

    def close(self) -> None:
        try:
            if self._dds_on:
                self.dds_stop()
        except Exception:
            pass
        try:
            self._dll_cookie.close() # type: ignore
        except Exception:
            pass
    
    def __enter__(self) -> "HantekHardDll":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
    
    def configure(self, params : ScanParams) -> None:
        time_div_idx = parse_time_div_idx(params.time_div)
        rx_volt_div_idx = parse_volt_div_idx(params.rx_volt_div)
        ref_volt_div_idx = (
            parse_volt_div_idx(params.ref_volt_div)
            if params.ref_channel is not None
            else None
        )

        scan_key = (
            time_div_idx,

            int(params.rx_channel),
            rx_volt_div_idx,
            int(params.rx_coupling),
            bool(params.rx_bw_limit),
            int(params.rx_lever_pos),

            int(params.ref_channel) if params.ref_channel is not None else -1,
            ref_volt_div_idx if ref_volt_div_idx is not None else -1,
            int(params.ref_coupling),
            bool(params.ref_bw_limit),
            int(params.ref_lever_pos),

            int(params.trig_source),
            int(params.trig_mode),
            int(params.trig_slope),
            int(params.trig_level_pos),
            int(params.read_len),
            int(params.pretrigger_percent),
            int(params.yt_format),
            int(params.start_control),
        )

        self._scan_key = scan_key
        self._time_div_idx = time_div_idx
        self._rx_volt_div_idx = rx_volt_div_idx
        self._ref_volt_div_idx = ref_volt_div_idx
        self.params = params

        self._ensure_buffers(params.read_len)
        self._fill_structs(params)

        _check_ok("dsoHTSetSampleRate",
            self.lib.dsoHTSetSampleRate(
                WORD(self.device_index),
                WORD(int(params.yt_format)),
                ct.byref(self._relay),
                ct.byref(self._ctrl),
            ))

        _check_ok("dsoHTSetCHAndTrigger",
            self.lib.dsoHTSetCHAndTrigger(
                WORD(self.device_index),
                ct.byref(self._relay),
                WORD(time_div_idx),
            ))

        _check_ok("dsoHTSetRamAndTrigerControl",
            self.lib.dsoHTSetRamAndTrigerControl(
                WORD(self.device_index),
                WORD(time_div_idx),
                WORD(int(self._ctrl.nCHSet)),
                WORD(int(params.trig_source)),
                WORD(0),
            ))

        # receiver channel position
        _check_ok(
            "dsoHTSetCHPos",
            self.lib.dsoHTSetCHPos(
                WORD(self.device_index),
                WORD(rx_volt_div_idx),
                WORD(params.rx_lever_pos),
                WORD(int(params.rx_channel)),
                WORD(self.channel_mode),
            )
        )

        # reference channel position
        if params.ref_channel is not None and ref_volt_div_idx is not None:
            _check_ok(
                "dsoHTSetCHPos",
                self.lib.dsoHTSetCHPos(
                    WORD(self.device_index),
                    WORD(ref_volt_div_idx),
                    WORD(params.ref_lever_pos),
                    WORD(int(params.ref_channel)),
                    WORD(self.channel_mode),
                )
            )

        _check_ok(
            "dsoHTSetVTriggerLevel",
            self.lib.dsoHTSetVTriggerLevel(
                WORD(self.device_index),
                WORD(params.trig_level_pos),
                WORD(params.trigger_sensitivity),
            )
        )

        _check_ok(
            "dsoHTSetTrigerMode",
            self.lib.dsoHTSetTrigerMode(
                WORD(self.device_index),
                WORD(int(params.trig_mode)),
                WORD(int(params.trig_slope)),
                WORD(int(params.trigger_coupling)),
            )
        )
        
        _check_ok(
            "dsoHTSetHTriggerLength",
            self.lib.dsoHTSetHTriggerLength(
                WORD(self.device_index),
                ct.byref(self._ctrl),
                WORD(self.channel_mode),
            )
        )

        ampcal_key = (
            self.channel_mode,
            int(self._ctrl.nCHSet),
            time_div_idx,
            rx_volt_div_idx,
            ref_volt_div_idx if ref_volt_div_idx is not None else -1,
            params.rx_lever_pos,
            params.ref_lever_pos if params.ref_channel is not None else -1,
        )

        if self._ampcal_key != ampcal_key:
            # массив volt-div на 4 канала (у вас он одинаковый для всех, но SDK ждёт массив)
            volt_arr = (WORD * MAX_CH_NUM)(
                WORD(rx_volt_div_idx), WORD(rx_volt_div_idx), WORD(rx_volt_div_idx), WORD(rx_volt_div_idx)
            )

            # массив позиций каналов (CH1 = lever_pos, остальные обычно 128)
            pos_arr = (WORD * MAX_CH_NUM)(
                WORD(128), WORD(128), WORD(128), WORD(128)
            )

            # RX channel
            volt_arr[int(params.rx_channel)] = WORD(rx_volt_div_idx)
            pos_arr[int(params.rx_channel)] = WORD(params.rx_lever_pos)

            # REF channel
            if params.ref_channel is not None and ref_volt_div_idx is not None:
                volt_arr[int(params.ref_channel)] = WORD(ref_volt_div_idx)
                pos_arr[int(params.ref_channel)] = WORD(params.ref_lever_pos)

            _check_ok(
                "dsoHTSetAmpCalibrate",
                self.lib.dsoHTSetAmpCalibrate(
                    WORD(self.device_index),
                    WORD(int(self._ctrl.nCHSet)),
                    WORD(time_div_idx),
                    volt_arr,
                    pos_arr,
                )
            )
            self._ampcal_key = ampcal_key

        # Номинальные коэффициенты по текущей ручке канала
        self._rx_nominal_zero_code_abs = float(MAX_DATA - params.rx_lever_pos)
        self._rx_nominal_scale_v_per_code = float(VOLT_DIV_V[rx_volt_div_idx]) / 32.0

        if ref_volt_div_idx is not None:
            self._ref_nominal_zero_code_abs = float(MAX_DATA - params.ref_lever_pos)
            self._ref_nominal_scale_v_per_code = float(VOLT_DIV_V[ref_volt_div_idx]) / 32.0
        else:
            self._ref_nominal_zero_code_abs = 0.0
            self._ref_nominal_scale_v_per_code = 0.0

        # Активные (калиброванные, если заданы, иначе номинальные)
        if params.rx_cal is not None:
            self._rx_zero_code_abs = float(params.rx_cal.zero_code_abs)
            self._rx_scale_v_per_code = float(params.rx_cal.scale_v_per_code)
        else:
            self._rx_zero_code_abs = self._rx_nominal_zero_code_abs
            self._rx_scale_v_per_code = self._rx_nominal_scale_v_per_code

        if params.ref_channel is not None:
            if params.ref_cal is not None:
                self._ref_zero_code_abs = float(params.ref_cal.zero_code_abs)
                self._ref_scale_v_per_code = float(params.ref_cal.scale_v_per_code)
            else:
                self._ref_zero_code_abs = self._ref_nominal_zero_code_abs
                self._ref_scale_v_per_code = self._ref_nominal_scale_v_per_code
        else:
            self._ref_zero_code_abs = 0.0
            self._ref_scale_v_per_code = 0.0

        # Фактическая fs (после set_sample_rate)
        self.fs_hz = float(self.lib.dsoGetSampleRate(WORD(self.device_index)))
        self.dt_s = 1.0 / self.fs_hz if self.fs_hz > 0 else 0.0

        if self.fs_hz <= 0:
            raise HantekError("Sample rate is zero after configure(); check time_div_idx / device state")
        
        # Кэш оси времени
        self._t_cache = self._build_time_axis(
            n=params.read_len,
            dt=self.dt_s,
            pretrigger_percent=params.pretrigger_percent,
        )

    def capture(
        self,
        *,
        point : Optional[Tuple[float, float]] = None,
        return_mode : Literal["raw", "volts", "both"] = "volts",
        copy : bool = False,
        apply_vert_cal : bool = True,
    ) -> AScanFrame:

        if self.params is None or self._t_cache is None:
            raise RuntimeError("Call configure(params) before capture()")

        p = self.params

        _check_ok("dsoHTStartCollectData",
            self.lib.dsoHTStartCollectData(WORD(self.device_index), WORD(int(p.start_control))))

        triggered = self._wait_capture_done(timeout_s=p.timeout_s, poll_s=p.poll_s)

        return self._read_frame_from_buffers(
            point=point,
            triggered=triggered,
            return_mode=return_mode,
            copy=copy,
            apply_vert_cal=apply_vert_cal,
        )
    
    def capture_burst(
        self,
        *,
        point : Optional[Tuple[float, float]] = None,
        return_mode : Literal["raw", "volts", "both"] = "volts",
        copy : bool = False,
        auto_enable_dds : bool = True,
        settle_after_enable_s : float = 0.0,
        arm_settle_s : float = 0.001,
        retries : int = 2,
        apply_vert_cal : bool = True,
    ) -> AScanFrame:
        if self.params is None or self._t_cache is None:
            raise RuntimeError("Call configure(scan_params) before capture_burst()")
        if self._dds_params is None:
            raise RuntimeError("Call dds_configure(dds_params) before capture_burst()")
        if self._dds_params.mode != DDSMode.BURST:
            raise RuntimeError("capture_burst() requires DDS mode BURST")

        p = self.params

        if auto_enable_dds and not self._dds_on:
            self.dds_output(True)
            if settle_after_enable_s > 0.0:
                time.sleep(settle_after_enable_s)

        last_exc : Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                _check_ok(
                    "dsoHTStartCollectData",
                    self.lib.dsoHTStartCollectData(
                        WORD(self.device_index),
                        WORD(int(p.start_control))
                    )
                )

                if arm_settle_s > 0.0:
                    time.sleep(arm_settle_s)

                self.dds_emit_single()

                triggered = self._wait_capture_done(timeout_s=p.timeout_s, poll_s=p.poll_s)

                return self._read_frame_from_buffers(
                    point=point,
                    triggered=triggered,
                    return_mode=return_mode,
                    copy=copy,
                    apply_vert_cal=apply_vert_cal,
                )
            except TimeoutError as e:
                last_exc = e
                if attempt >= retries:
                    raise
                # short pause before re-arming
                time.sleep(0.01)
        assert last_exc is not None
        raise last_exc

    def _wait_capture_done(self, timeout_s : float, poll_s : float) -> bool:
        triggered = False
        t0 = time.perf_counter()
        last_raw_state = 0

        while True:
            raw_state = int(self.lib.dsoHTGetState(WORD(self.device_index)))
            last_raw_state = raw_state

            # documented bits are only bit0 and bit1
            state_bits = raw_state & 0x0003

            if state_bits & int(DeviceState.TRIGGERED):
                triggered = True
            if state_bits & int(DeviceState.DONE):
                return triggered

            if time.perf_counter() - t0 >= timeout_s:
                raise TimeoutError(
                    f"Capture timeout {timeout_s}s; raw_state={last_raw_state:#06x}; "
                    f"state_bits={state_bits:#04x}"
                )
            time.sleep(poll_s)

    def _read_frame_from_buffers(
        self,
        *,
        point : Optional[Tuple[float, float]],
        triggered : bool,
        return_mode : Literal["raw", "volts", "both"],
        copy : bool,
        apply_vert_cal : bool = True,
    ) -> AScanFrame:
        assert self._ch1_buf is not None
        assert self._ch2_buf is not None
        assert self._ch3_buf is not None
        assert self._ch4_buf is not None
        assert self._t_cache is not None

        p = self.params
        assert p is not None

        _check_ok(
            "dsoHTGetData",
            self.lib.dsoHTGetData(
                WORD(self.device_index),
                self._ch1_buf, self._ch2_buf, self._ch3_buf, self._ch4_buf,
                ct.byref(self._ctrl),
            )
        )

        n = int(self._ctrl.nReadDataLen)
        rx_buf = self._buffer_for_channel(p.rx_channel)
        assert rx_buf is not None
        rx_raw_view = np.ctypeslib.as_array(rx_buf)[:n]

        ref_raw_view = None
        if p.ref_channel is not None:
            ref_buf = self._buffer_for_channel(p.ref_channel)
            assert ref_buf is not None
            ref_raw_view = np.ctypeslib.as_array(ref_buf)[:n]

        t_out = self._t_cache[:n].copy() if copy else self._t_cache[:n]

        rx_volts_out : Optional[np.ndarray] = None
        ref_volts_out : Optional[np.ndarray] = None
        rx_raw_out : Optional[np.ndarray] = None
        ref_raw_out : Optional[np.ndarray] = None

        if return_mode in {"raw", "both"}:
            rx_raw_out = rx_raw_view.copy() if copy else rx_raw_view
            if ref_raw_view is not None:
                ref_raw_out = ref_raw_view.copy() if copy else ref_raw_view

        if return_mode in {"volts", "both"}:
            if apply_vert_cal:
                rx_zero_code_abs = self._rx_zero_code_abs
                rx_scale_v_per_code = self._rx_scale_v_per_code
            else:
                rx_zero_code_abs = self._rx_nominal_zero_code_abs
                rx_scale_v_per_code = self._rx_nominal_scale_v_per_code

            rx_centered_codes = raw_to_centered_codes(
                rx_raw_view,
                zero_code_abs=rx_zero_code_abs,
            )
            rx_volts_view = codes_to_volts(
                rx_centered_codes,
                scale_v_per_code=rx_scale_v_per_code,
            )
            rx_volts_out = rx_volts_view.copy() if copy else rx_volts_view

            if ref_raw_view is not None:
                if apply_vert_cal:
                    ref_zero_code_abs = self._ref_zero_code_abs
                    ref_scale_v_per_code = self._ref_scale_v_per_code
                else:
                    ref_zero_code_abs = self._ref_nominal_zero_code_abs
                    ref_scale_v_per_code = self._ref_nominal_scale_v_per_code

                ref_centered_codes = raw_to_centered_codes(
                    ref_raw_view,
                    zero_code_abs=ref_zero_code_abs,
                )
                ref_volts_view = codes_to_volts(
                    ref_centered_codes,
                    scale_v_per_code=ref_scale_v_per_code,
                )
                ref_volts_out = ref_volts_view.copy() if copy else ref_volts_view

        rx_overflow = is_overflow(rx_raw_view)
        ref_overflow = is_overflow(ref_raw_view) if ref_raw_view is not None else False

        return AScanFrame(
            point=point,
            fs_hz=self.fs_hz,
            dt_s=self.dt_s,
            t_s=t_out,
            triggered=triggered,
            rx_volts=rx_volts_out,
            rx_raw_u16=rx_raw_out,
            ref_volts=ref_volts_out,
            ref_raw_u16=ref_raw_out,
            rx_overflow=rx_overflow,
            ref_overflow=ref_overflow,
        )

    def _find_device(self) -> int:
        for idx in range(self.max_devices):
            if self.lib.dsoHTDeviceConnect(WORD(idx)):
                return idx
        raise HantekError("Device not found (dsoHTDeviceConnect==0 for all indices)")

    def _ensure_buffers(self, n : int) -> None:
        if self._buf_len >= n and self._ch1_buf is not None:
            return
        self._buf_len = n
        self._ch1_buf = (WORD * n)()
        self._ch2_buf = (WORD * n)()
        self._ch3_buf = (WORD * n)()
        self._ch4_buf = (WORD * n)()

    def _fill_structs(self, p : ScanParams) -> None:
        assert self._time_div_idx is not None
        assert self._rx_volt_div_idx is not None

        enabled_channels = {int(p.rx_channel)}
        if p.ref_channel is not None:
            enabled_channels.add(int(p.ref_channel))

        for ch in range(MAX_CH_NUM):
            enabled = 1 if ch in enabled_channels else 0
            self._relay.bCHEnable[ch] = enabled

            if ch == int(p.rx_channel):
                self._relay.nCHVoltDIV[ch] = WORD(self._rx_volt_div_idx)
                self._relay.nCHCoupling[ch] = WORD(int(p.rx_coupling))
                self._relay.bCHBWLimit[ch] = 1 if p.rx_bw_limit else 0
            elif p.ref_channel is not None and ch == int(p.ref_channel):
                assert self._ref_volt_div_idx is not None
                self._relay.nCHVoltDIV[ch] = WORD(self._ref_volt_div_idx)
                self._relay.nCHCoupling[ch] = WORD(int(p.ref_coupling))
                self._relay.bCHBWLimit[ch] = 1 if p.ref_bw_limit else 0
            else:
                self._relay.nCHVoltDIV[ch] = WORD(self._rx_volt_div_idx)
                self._relay.nCHCoupling[ch] = WORD(int(p.rx_coupling))
                self._relay.bCHBWLimit[ch] = 0

        self._relay.nTrigSource = WORD(int(p.trig_source))
        self._relay.bTrigFilt = 0
        self._relay.nALT = 0

        # ControlData
        ch_mask = 0
        for ch in enabled_channels:
            ch_mask |= (1 << ch)

        self._ctrl.nCHSet = WORD(ch_mask)
        self._ctrl.nTimeDIV = WORD(self._time_div_idx)
        self._ctrl.nTriggerSource = WORD(int(p.trig_source))
        self._ctrl.nHTriggerPos = WORD(p.pretrigger_percent)
        self._ctrl.nVTriggerPos = WORD(p.trig_level_pos)
        self._ctrl.nTriggerSlope = WORD(int(p.trig_slope))
        self._ctrl.nBufferLen = UINT(p.read_len)
        self._ctrl.nReadDataLen = UINT(p.read_len)
        self._ctrl.nAlreadyReadLen = UINT(0)
        self._ctrl.nALT = 0
        self._ctrl.nETSOpen = 0
        self._ctrl.nDriverCode = 0
        self._ctrl.nLastAddress = 0
        self._ctrl.nFPGAVersion = 0

    def _buffer_for_channel(self, ch : Channel) -> ct.Array[WORD] | None:
        if ch == Channel.CH1:
            return self._ch1_buf
        if ch == Channel.CH2:
            return self._ch2_buf
        if ch == Channel.CH3:
            return self._ch3_buf
        if ch == Channel.CH4:
            return self._ch4_buf
        raise ValueError(f"Unsupported channel: {ch}")

    @staticmethod
    def _build_time_axis(*, n : int, dt : float, pretrigger_percent : int) -> np.ndarray:
        trig_idx = int(round(n * (pretrigger_percent / 100.0)))
        return (np.arange(n, dtype=np.float64) - trig_idx) * dt
    
    # ============================================================
    # DDS helpers
    # ============================================================

    def _dds_make_key(self, params : DDSParams) -> Tuple:
        return (
            float(params.frequency_hz),
            int(params.amplitude_mv),
            int(params.offset_mv),
            int(params.wave_type),
            float(params.phase_frac),
            float(params.duty),
            int(params.burst_cycles),
            int(params.mode),
        )

    def dds_output(self, on : bool) -> None:
        """Включить/выключить генератор"""
        val = 1 if on else 0
        _check_ok("ddsSetOnOff", self.lib.ddsSetOnOff(WORD(self.device_index), ct.c_short(val)))
        self._dds_on = on

    def dds_stop(self) -> None:
        self.dds_output(False)

    def dds_configure(self, params : DDSParams, *, force : bool = False) -> None:
        key = self._dds_make_key(params)
        if (not force) and (self._dds_key == key):
            return

        _check_ok("ddsSetCmd", self.lib.ddsSetCmd(WORD(self.device_index), WORD(params.mode)))

        self.lib.ddsSDKSetFre(WORD(self.device_index), ct.c_float(params.frequency_hz))
        self.lib.ddsSDKSetAmp(WORD(self.device_index), WORD(params.amplitude_mv))
        self.lib.ddsSDKSetOffset(WORD(self.device_index), ct.c_short(params.offset_mv))
        self.lib.ddsSDKSetWaveType(WORD(self.device_index), WORD(int(params.wave_type)))
        self.lib.ddsSDKSetWavePhase(WORD(self.device_index), ct.c_float(params.phase_frac))

        if params.wave_type == DDSWaveType.SQUARE:
            self.lib.ddsSDKSetWaveDuty(WORD(self.device_index), ct.c_float(params.duty))

        if params.mode == DDSMode.BURST:
            self.lib.ddsSDKSetBurstNum(WORD(self.device_index), WORD(params.burst_cycles))

        self._dds_params = params
        self._dds_key = key

    def dds_emit_single(self) -> None:
        """
        Выдать одиночный burst / single-shot wave
        Использовать только если DDS настроен в burst/single режиме
        """
        if self._dds_params is None:
            raise RuntimeError("Call dds_configure(...) before dds_emit_single()")
        if self._dds_params.mode != DDSMode.BURST:
            raise RuntimeError("dds_emit_single() requires DDS mode BURST")
        _check_ok("ddsEmitSingle", self.lib.ddsEmitSingle(WORD(self.device_index)))