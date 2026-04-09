# hantek_scope

Python API for Hantek 6000BD-series USB oscilloscopes and built-in DDS generator

> Status: early-stage hardware-facing library
> Tested workflow: Hantek 6074BD on Windows with the vendor 6000BD SDK
> Intended family: Hantek 6074BD / 6104BD / 6204BD / 6254BD, provided that the vendor SDK and DLL interface are compatible

`hantek_scope` is a thin but practical Python API around the vendor `Hantek 6000BD SDK` for Windows

It is designed for scripted acquisition, burst-driven measurements, synchronized averaging, and experimental automation workflows where the oscilloscope is controlled from Python instead of the vendor GUI

The package focuses on:
- explicit acquisition configuration
- raw-code and voltage-domain capture
- optional vertical calibration
- burst acquisition through the built-in DDS generator
- synchronized averaging with optional time rebasing to a reference channel
- gate-based signal analysis utilities

## Why this project exists

The vendor software is convenient for interactive work, but it is poorly suited for:
- repeatable laboratory scripts
- integration with scanners, robots, or motion systems
- bulk acquisition and averaging
- signal post-processing directly in Python
- reproducible experimental pipelines

This repository provides a small, explicit Python API for those workflows

## Scope

This repository contains:
- Python bindings to the vendor DLL
- typed configuration models for acquisition and DDS settings
- high-level capture helpers
- synchronized averaging utilities
- time-gating helpers
- optional vertical calibration helpers

This repository does **not** include:
- the vendor SDK itself
- the vendor DLL binaries
- the vendor GUI software

You must download the vendor SDK separately from the official Hantek website

## Supported hardware and platform

According to the official Hantek product/download page for the 6000BD family, the relevant model family includes:
- Hantek6074BD
- Hantek6104BD
- Hantek6204BD
- Hantek6254BD

The current Python implementation is **Windows-only**, because it loads the vendor `HTHardDll.dll` via `ctypes.WinDLL` and uses the Windows DLL search path mechanism

If you use a different device from the same family, treat it as **supported by design, but not guaranteed by this repository unless tested**

## Vendor SDK installation

The vendor SDK is **not** bundled with this repository
You must download it separately from the official Hantek page for the 6000BD family, open the **Download** section, and download **Hantek6000BD SDK**

Official page:

https://www.hantek.com/products/detail/10164

After extracting the SDK, this project expects the DLL directory to be resolved **via `get_default_dll_dir()`** from `hantek_scope.paths`

`get_default_dll_dir()` supports exactly two lookup modes:

### Option A — via environment variable

Set:
```powershell
$env:HANTEK_DLL_DIR="C:\path\to\Hantek_SDK\Dll\x64"
```

The directory must contain:
```text
HTHardDll.dll
```

### Option B — via the default repository-local path

Place the extracted SDK inside the repository like this:

```text
Hantek_SDK/
└── Dll/
    └── x64/
        └── HTHardDll.dll
```

If `HANTEK_DLL_DIR` is not set, `get_default_dll_dir()` will look for the SDK in this default location

All documented examples in this repository assume that the DLL path is obtained through:

```python
from hantek_scope import get_default_dll_dir
```

## Installation

### Create and activate a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install the package in editable mode

```powershell
py -m pip install --upgrade pip
py -m pip install -e .
```

### Install development and test dependencies

For package development, verification, and testing, install the additional dependencies listed in `requirements-dev.txt`:

```powershell
py -m pip install -r requirements-dev.txt
```

This file is intended for contributors and maintainers of `hantek_scope`
It is not required for basic runtime use of the package

## Quick start

### 1. Single acquisition

```python
from hantek_scope import HantekHardDll, make_scan_params_nominal, get_default_dll_dir

params = make_scan_params_nominal(
    rx_volt_div="500mV",
    ref_volt_div="1V",
    time_div="50us",
    read_len=0x4000,
    pretrigger_percent=10,
)

with HantekHardDll(get_default_dll_dir()) as scope:
    scope.configure(params)
    frame = scope.capture(return_mode="both", copy=True)

print(frame.fs_hz, frame.dt_s, frame.triggered)
print(frame.t_s.shape)
print(frame.rx_raw_u16.shape if frame.rx_raw_u16 is not None else None)
print(frame.rx_volts.shape if frame.rx_volts is not None else None)
```

### 2. Burst acquisition using the built-in generator

```python
from hantek_scope import (
    HantekHardDll,
    DDSParams,
    DDSMode,
    DDSWaveType,
    make_scan_params_nominal,
    get_default_dll_dir,
)

scan_params = make_scan_params_nominal(
    rx_volt_div="200mV",
    ref_volt_div="1V",
    time_div="20us",
    read_len=0x4000,
    pretrigger_percent=10,
)

dds_params = DDSParams(
    frequency_hz=125_000.0,
    amplitude_mv=3500,
    wave_type=DDSWaveType.SQUARE,
    duty=0.3,
    burst_cycles=1,
    mode=DDSMode.BURST,
)

with HantekHardDll(get_default_dll_dir()) as scope:
    scope.configure(scan_params)
    scope.dds_configure(dds_params)
    frame = scope.capture_burst(return_mode="both", copy=True)

print(frame.triggered, frame.rx_overflow, frame.ref_overflow)
```

### 3. Synchronized averaging

```python
from hantek_scope import (
    HantekHardDll,
    make_scan_params_nominal,
    capture_average_burst_fast,
    TimeGate,
    get_default_dll_dir,
)

params = make_scan_params_nominal(
    rx_volt_div="200mV",
    ref_volt_div="1V",
    time_div="20us",
    read_len=0x4000,
    pretrigger_percent=10,
)

ref_gate = TimeGate(t1_s=-3e-6, t2_s=10e-6, name="ref_gate")

with HantekHardDll(get_default_dll_dir()) as scope:
    scope.configure(params)

    res = capture_average_burst_fast(
        scope,
        K=200,
        repeat_s=0.01,
        rebasing="auto",
        edge_frac=0.5,
        ref_edge_gate=ref_gate,
        output_mode="codes",
    )

print(res.accepted_count)
print(res.used_rebasing)
print(res.ref_edge_std_samples)
print(res.rx_codes.shape if res.rx_codes is not None else None)
```

## Public API overview

### Main device class

* `HantekHardDll` — low-level device session:
  * device discovery
  * oscilloscope configuration
  * single acquisition via `capture(...)`
  * burst acquisition via `capture_burst(...)`
  * DDS control via `dds_configure(...)`, `dds_output(...)`, `dds_emit_single(...)`, `dds_stop(...)`

### Configuration models

* `ScanParams` — acquisition settings:

  * channels
  * vertical scale
  * couplings
  * trigger source and trigger mode
  * read length
  * pretrigger
  * optional vertical calibration

* `DDSParams` — built-in generator settings:

  * frequency
  * amplitude
  * offset
  * waveform type
  * duty cycle
  * burst cycle count
  * mode

* `VertCal` — vertical calibration coefficients

### Acquisition results

* `AScanFrame` — one captured frame
* `AveragedBurstResult` — averaged burst result
* `ScanPointPayload` — compact payload for one averaged scan point

### Helpers

* parsing:

  * `parse_time_div_idx`
  * `parse_volt_div_idx`

* calibration:

  * `VERT_CAL_DB`
  * `resolve_vert_cal`
  * `make_scan_params_from_db`
  * `make_scan_params_nominal`

* reference alignment:

  * `estimate_edge_index`
  * `rebase_time_to_reference`

* gating:

  * `TimeGate`
  * `gate_mask`
  * `gate_indices`
  * `extract_in_gate`
  * `crop_to_gate`
  * `peak_abs_in_gate`
  * `rms_in_gate`
  * `rms_ac_in_gate`
  * `argmax_abs_index_in_gate`

* averaging:

  * `capture_average_burst_fast`
  * `capture_scan_point_fast`

## Calibration model

The package supports two signal representations:

1. **raw ADC codes**
2. **voltage-domain data**

For voltage conversion, the package can use either:

- nominal conversion derived from the current vertical scale
- explicit calibration coefficients from `VERT_CAL_DB` in `hantek_scope.calibration`

If you do not yet have a validated calibration profile for your probe or acquisition path, start with nominal settings and describe that choice explicitly in your experiment documentation

The scripts in `scripts/calibration` are intended for **manual calibration workflow**:
- estimating vertical calibration coefficients for a specific acquisition path
- validating those coefficients
- updating the values stored in `VERT_CAL_DB`

These scripts are not required for basic acquisition
They are maintenance utilities for building and checking the calibration table used by `make_scan_params_from_db(...)`

## Error handling and operational notes

* The library raises `HantekError` for vendor-API failures
* Timeouts during acquisition raise `TimeoutError`
* Overflow flags are exposed on captured frames
* In synchronized averaging, frames may be skipped due to:

  * overflow
  * missing trigger
  * missing reference edge

This behavior is intentional and is designed to make burst-averaged acquisitions more robust in experimental conditions

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements-dev.txt
├── .gitignore
├── src/
│   └── hantek_scope/
│       ├── __init__.py
│       ├── acquisition.py
│       ├── averaging.py
│       ├── calibration.py
│       ├── constants.py
│       ├── device.py
│       ├── enums.py
│       ├── gating.py
│       ├── models.py
│       ├── parsing.py
│       ├── paths.py
│       ├── sdk_bindings.py
│       └── sdk_structs.py
└── scripts/
    ├── calibration/        # manual tools for filling and checking VERT_CAL_DB
    └── verification/
```

## Development notes

This project uses a `src/` layout, which is a good fit for distributable Python packages and helps avoid accidentally importing files from the repository root instead of the installed package

Recommended development workflow:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -e .
py -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` contains dependencies used for development, verification, and testing of hantek_scope
It should be kept under version control as part of the repository setup for contributors

## What this repository intentionally does not promise

This is **not** a full reimplementation of the vendor software

The project does not currently promise:

* GUI parity with the official Hantek application
* cross-platform support outside Windows
* support for arbitrary Hantek product families outside the 6000BD DLL interface
* stable compatibility with every future vendor SDK revision
* universally valid calibration profiles for every probe and acquisition setup

## Acknowledgements

This project depends on the official Hantek 6000BD SDK and the vendor DLL interface provided by Hantek