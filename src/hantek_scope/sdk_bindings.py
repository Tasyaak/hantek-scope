from __future__ import annotations
import os, ctypes as ct
from pathlib import Path
from .sdk_structs import WORD, RELAYCONTROL, CONTROLDATA


def load_hantek_library(dll_dir : Path) -> tuple[ct.WinDLL, object]:
    dll_dir = Path(dll_dir).resolve()
    dll_path = dll_dir / "HTHardDll.dll"

    if not dll_path.exists():
        raise FileNotFoundError(f"HTHardDll.dll not found: {dll_path}")

    dll_cookie = os.add_dll_directory(str(dll_dir))
    lib = ct.WinDLL(str(dll_path))
    return lib, dll_cookie


def bind_hantek_prototypes(lib : ct.WinDLL) -> None:
        # connect/find
        lib.dsoHTDeviceConnect.argtypes = [WORD]
        lib.dsoHTDeviceConnect.restype = WORD

        # init/config
        lib.dsoInitHard.argtypes = [WORD]
        lib.dsoInitHard.restype = WORD

        lib.dsoHTADCCHModGain.argtypes = [WORD, WORD]
        lib.dsoHTADCCHModGain.restype = WORD

        lib.dsoHTSetSampleRate.argtypes = [WORD, WORD, ct.POINTER(RELAYCONTROL), ct.POINTER(CONTROLDATA)]
        lib.dsoHTSetSampleRate.restype = WORD

        lib.dsoHTSetCHAndTrigger.argtypes = [WORD, ct.POINTER(RELAYCONTROL), WORD]
        lib.dsoHTSetCHAndTrigger.restype = WORD

        lib.dsoHTSetRamAndTrigerControl.argtypes = [WORD, WORD, WORD, WORD, WORD]
        lib.dsoHTSetRamAndTrigerControl.restype = WORD

        lib.dsoHTSetCHPos.argtypes = [WORD, WORD, WORD, WORD, WORD]
        lib.dsoHTSetCHPos.restype = WORD

        lib.dsoHTSetVTriggerLevel.argtypes = [WORD, WORD, WORD]
        lib.dsoHTSetVTriggerLevel.restype = WORD

        lib.dsoHTSetTrigerMode.argtypes = [WORD, WORD, WORD, WORD]
        lib.dsoHTSetTrigerMode.restype = WORD

        lib.dsoHTSetHTriggerLength.argtypes = [WORD, ct.POINTER(CONTROLDATA), WORD]
        lib.dsoHTSetHTriggerLength.restype = WORD

        lib.dsoHTReadCalibrationData.argtypes = [WORD, ct.POINTER(WORD), WORD]
        lib.dsoHTReadCalibrationData.restype = WORD

        lib.dsoHTSetAmpCalibrate.argtypes = [WORD, WORD, WORD, ct.POINTER(WORD), ct.POINTER(WORD)]
        lib.dsoHTSetAmpCalibrate.restype = WORD

        # acquisition
        lib.dsoHTStartCollectData.argtypes = [WORD, WORD]
        lib.dsoHTStartCollectData.restype = WORD

        lib.dsoHTGetState.argtypes = [WORD]
        lib.dsoHTGetState.restype = WORD

        lib.dsoHTGetData.argtypes = [
            WORD,
            ct.POINTER(WORD), ct.POINTER(WORD), ct.POINTER(WORD), ct.POINTER(WORD),
            ct.POINTER(CONTROLDATA),
        ]
        lib.dsoHTGetData.restype = WORD

        # sample rate
        lib.dsoGetSampleRate.argtypes = [WORD]
        lib.dsoGetSampleRate.restype = ct.c_float

        # DDS
        lib.ddsSetOnOff.argtypes = [WORD, ct.c_short]
        lib.ddsSetOnOff.restype = ct.c_ulong

        lib.ddsSetCmd.argtypes = [WORD, WORD]
        lib.ddsSetCmd.restype = ct.c_ulong

        lib.ddsEmitSingle.argtypes = [WORD]
        lib.ddsEmitSingle.restype = ct.c_ulong

        lib.ddsSDKSetFre.argtypes = [WORD, ct.c_float]
        lib.ddsSDKSetFre.restype = ct.c_float

        lib.ddsSDKSetAmp.argtypes = [WORD, WORD]
        lib.ddsSDKSetAmp.restype = WORD

        lib.ddsSDKSetOffset.argtypes = [WORD, ct.c_short]
        lib.ddsSDKSetOffset.restype = ct.c_short

        lib.ddsSDKSetBurstNum.argtypes = [WORD, WORD]
        lib.ddsSDKSetBurstNum.restype = WORD

        lib.ddsSDKSetWaveType.argtypes = [WORD, WORD]
        lib.ddsSDKSetWaveType.restype = WORD

        lib.ddsSDKSetWavePhase.argtypes = [WORD, ct.c_float]
        lib.ddsSDKSetWavePhase.restype = ct.c_float

        lib.ddsSDKSetWaveDuty.argtypes = [WORD, ct.c_float]
        lib.ddsSDKSetWaveDuty.restype = ct.c_float