import ctypes as ct
from ctypes import wintypes
from .constants import MAX_CH_NUM


WORD = wintypes.WORD    # uint16
UINT = wintypes.UINT    # uint32
BOOL = wintypes.BOOL


class RELAYCONTROL(ct.Structure):
    _fields_ = [
        ("bCHEnable", BOOL * MAX_CH_NUM),
        ("nCHVoltDIV", WORD * MAX_CH_NUM),
        ("nCHCoupling", WORD * MAX_CH_NUM),
        ("bCHBWLimit", BOOL * MAX_CH_NUM),
        ("nTrigSource", WORD),
        ("bTrigFilt", BOOL),
        ("nALT", WORD),
    ]


class CONTROLDATA(ct.Structure):
    _fields_ = [
        ("nCHSet", WORD),
        ("nTimeDIV", WORD),
        ("nTriggerSource", WORD),
        ("nHTriggerPos", WORD),     # pretrigger %, 0..100
        ("nVTriggerPos", WORD),     # trigger level pos, 0..255 (условно)
        ("nTriggerSlope", WORD),
        ("nBufferLen", UINT),
        ("nReadDataLen", UINT),
        ("nAlreadyReadLen", UINT),
        ("nALT", WORD),
        ("nETSOpen", WORD),
        ("nDriverCode", WORD),
        ("nLastAddress", UINT),
        ("nFPGAVersion", WORD),
    ]