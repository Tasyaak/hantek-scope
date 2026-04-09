from decimal import Decimal


MAX_CH_NUM = 4
MAX_DATA   = 255


TIME_DIV_S = [
    # ns
    Decimal("2e-9"), Decimal("5e-9"), Decimal("1e-8"),
    Decimal("2e-8"), Decimal("5e-8"), Decimal("1e-7"),
    Decimal("2e-7"), Decimal("5e-7"),
    # us
    Decimal("1e-6"), Decimal("2e-6"), Decimal("5e-6"),
    Decimal("1e-5"), Decimal("2e-5"), Decimal("5e-5"),
    Decimal("1e-4"), Decimal("2e-4"), Decimal("5e-4"),
    # ms
    Decimal("1e-3"), Decimal("2e-3"), Decimal("5e-3"),
    Decimal("1e-2"), Decimal("2e-2"), Decimal("5e-2"),
    Decimal("1e-1"), Decimal("2e-1"), Decimal("5e-1"),
    # s
    Decimal("1"), Decimal("2"), Decimal("5"),
    Decimal("1e1"), Decimal("2e1"), Decimal("5e1"),
    Decimal("1e2"), Decimal("2e2"), Decimal("5e2"),
    Decimal("1e3"),
]
TIME_DIV_STR = [
    "2ns","5ns","10ns","20ns","50ns","100ns","200ns","500ns",
    "1us","2us","5us","10us","20us","50us","100us","200us","500us",
    "1ms","2ms","5ms","10ms","20ms","50ms","100ms","200ms","500ms",
    "1s","2s","5s","10s","20s","50s","100s","200s","500s","1000s"
]


VOLT_DIV_V = [
    # mV
    Decimal("2e-3"), Decimal("5e-3"), Decimal("1e-2"),
    Decimal("2e-2"), Decimal("5e-2"), Decimal("1e-1"),
    Decimal("2e-1"), Decimal("5e-1"),
    # V
    Decimal("1"), Decimal("2"), Decimal("5"),
    Decimal("1e1"),
]
VOLT_DIV_STR = [
    "2mV","5mV","10mV","20mV","50mV","100mV","200mV","500mV",
    "1V","2V","5V","10V"
]