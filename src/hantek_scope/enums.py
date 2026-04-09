from enum import IntEnum, IntFlag


class Channel(IntEnum):
    CH1 = 0
    CH2 = 1
    CH3 = 2
    CH4 = 3


class Coupling(IntEnum):
    DC  = 0
    AC  = 1
    GND = 2


class TriggerMode(IntEnum):
    EDGE = 0


class TriggerSlope(IntEnum):
    RISE = 0
    FALL = 1


class TriggerCoupling(IntEnum):
    DC = 0
    AC = 1
    LF_REJECT    = 2
    HF_REJECT    = 3
    NOISE_REJECT = 4


class YTFormat(IntEnum):
    NORMAL = 0


class StartControl(IntFlag):
    """
    Управляющие биты для dsoHTStartCollectData:
      bit0: AUTO trigger
      bit1: ROLL mode
      bit2: stop after this collect
    """
    NONE       = 0
    AUTO       = 1 << 0
    ROLL       = 1 << 1
    STOP_AFTER = 1 << 2


class DeviceState(IntFlag):
    """
    Биты, возвращаемые dsoHTGetState:
      bit0: триггер уже сработал
      bit1: сбор данных завершён
    """
    TRIGGERED = 1 << 0
    DONE      = 1 << 1


class DDSWaveType(IntEnum):
    SINE   = 0
    RAMP   = 1
    SQUARE = 2
    DC     = 4
    NOISE  = 8


class DDSMode(IntEnum):
    CONTINUOUS = 0
    BURST      = 4