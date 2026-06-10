import re
from decimal import Decimal, InvalidOperation
from .constants import TIME_DIV_S, VOLT_DIV_V, TIME_DIV_STR, VOLT_DIV_STR


_TIME_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(ns|us|ms|s)\s*$", re.IGNORECASE)
_VOLT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(mv|v)\s*$", re.IGNORECASE)


def _dec(x : str) -> Decimal:
    try:
        return Decimal(x)
    except InvalidOperation as e:
        raise ValueError(f"Bad number: {x}") from e


def parse_time_div_idx(value : int | str) -> int:
    """
    value: int (готовый индекс) или строка вроде '10ms', '2ns', '0.5s'
    Возвращает точный idx, если значения нету в таблице — ValueError
    """
    if isinstance(value, bool):
        raise TypeError(f"time_div_idx must be int or str, got: bool")
    
    if isinstance(value, int):
        if 0 <= value < len(TIME_DIV_S):
            return value
        raise ValueError(f"time_div_idx out of range: {value}")

    if not isinstance(value, str):
        raise TypeError(f"time_div must be int or str, got: {type(value).__name__}")

    m = _TIME_RE.match(value)
    if not m:
        raise ValueError(f"Bad time_div format: {value!r} (use e.g. '10ms', '2ns', '1s')")

    num = _dec(m.group(1))
    unit = m.group(2).lower()

    scale = {"ns": Decimal("1e-9"), "us": Decimal("1e-6"), "ms": Decimal("1e-3"), "s": Decimal("1")}[unit]
    v = (num * scale)

    try:
        return TIME_DIV_S.index(v)
    except ValueError as e:
        raise ValueError(f"Unsupported time_div value: {value!r}. Allowed: {TIME_DIV_STR}") from e


def parse_volt_div_idx(value : int | str) -> int:
    """
    value: int (готовый индекс) или строка вроде '1V', '100mV', '2mV'
    Возвращает точный idx. Если значения нет в таблице — ValueError
    """
    if isinstance(value, bool):
        raise TypeError(f"volt_div_idx must be int or str, got: bool")
    
    if isinstance(value, int):
        if 0 <= value < len(VOLT_DIV_V):
            return value
        raise ValueError(f"volt_div_idx out of range: {value}")

    if not isinstance(value, str):
        raise TypeError(f"volt_div must be int or str, got: {type(value).__name__}")

    m = _VOLT_RE.match(value)
    if not m:
        raise ValueError(f"Bad volt_div format: {value!r} (use e.g. '1V', '200mV', '2mV')")

    num = _dec(m.group(1))
    unit = m.group(2).lower()

    scale = {"mv": Decimal("1e-3"), "v": Decimal("1")}[unit]
    v = (num * scale)

    try:
        return VOLT_DIV_V.index(v)
    except ValueError as e:
        raise ValueError(f"Unsupported volt_div value: {value!r}. Allowed: {VOLT_DIV_STR}") from e