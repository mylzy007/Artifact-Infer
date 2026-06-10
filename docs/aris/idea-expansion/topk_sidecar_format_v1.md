# Topk Sidecar Binary Format v1

- Status: **v1 frozen** (any breaking change must bump the version byte)
- Date: 2026-05-29
- Author: Stage D0 (D1-I10 / D2-I5 prep)
- Companion: jsonl file `per_step_trace_<run_id>_rank<r>.jsonl`
- Sidecar file: `per_step_trace_<run_id>_rank<r>_topk.bin`
- One sidecar **per (run_id, rank)** — matches jsonl 1:1.

## Endianness
**Little-endian throughout.** All multi-byte integers are little-endian. All `float16` values are IEEE 754 binary16 little-endian (same as `numpy.dtype('<f2')`).

## Numpy dtype equivalents (canonical)
- `int8`  ≡ `numpy.dtype('<i1')` (signed; expert ids 0..127 fit positive range)
- `float16` ≡ `numpy.dtype('<f2')`
- `uint32` ≡ `numpy.dtype('<u4')`
- `uint8`  ≡ `numpy.dtype('u1')`

## File-level header (8 bytes, at file offset 0)
| offset | size | type | name | value (v1) |
|---|---|---|---|---|
| 0  | 4 | bytes  | `magic`    | `b'TKBN'` |
| 4  | 1 | uint8  | `version`  | `1`       |
| 5  | 1 | uint8  | `K`        | `8` for Qwen3-MoE (router top-K), constant for the whole file |
| 6  | 2 | uint8×2 | `reserved` | `0x00 0x00` (must be zero in v1) |

A reader MUST:
- Verify `magic == b'TKBN'` and `version == 1`.
- Use the file-level `K` for all records in this file (records do **not** repeat K — per user spec).

## Records
- Records are written **back-to-back, no padding, no inter-record magic.**
- First record starts at byte offset **8** (= end of file header).
- Record layout (variable length, total bytes = `12 + T*K*3`):

| offset (within record) | size | type | name |
|---|---|---|---|
| 0  | 4   | uint32 LE | `step_id`   |
| 4  | 4   | uint32 LE | `layer_id`  |
| 8  | 4   | uint32 LE | `T` (num_tokens this rank) |
| 12 | T·K | int8[T·K] | `topk_ids` (flattened, row-major over tokens, then K-axis) |
| 12 + T·K | 2·T·K | float16[T·K] LE | `topk_weights` (same ordering as ids) |

- `topk_ids[t*K + k]` = global expert id (0..E_global-1) chosen as the k-th top expert for the t-th local token. Stored as **signed int8** because E_global ≤ 128 fits the positive range; negative values are unused in v1.
- `topk_weights[t*K + k]` = post-softmax post-norm router weight for the same (t, k) replica. Rounded to float16 precision before write.
- Both arrays are dense (no nan / no sentinel) when `T > 0`. When `T == 0`, no record is written and the corresponding jsonl record carries `topk_offset = -1`.

## JSONL companion fields (added to each record when sidecar is enabled)
| field | type | meaning |
|---|---|---|
| `topk_offset` | int | byte offset of the record's first byte (the `step_id` u32) within the sidecar binary. `-1` when `T == 0` (no record written). |
| `topk_T`      | int | per-record `T`. Mirrors record header to allow validation without reading binary. |
| `topk_K`      | int | per-record `K`. Mirrors file-level `K`; same value for every record in v1. |

When the sidecar is NOT in use (default opt-in `MOE_COLLECT_PER_STEP=1` with no topk capture), these three fields are absent and the jsonl is byte-compatible with GATE C.

## Hardening (collision check)
The same `MOE_COLLECT_FORCE_APPEND` semantics as the jsonl apply to the sidecar:
- Default: if `per_step_trace_<run_id>_rank<r>_topk.bin` already exists, recorder **raises `FileExistsError`**.
- `MOE_COLLECT_FORCE_APPEND=1`: opens in `ab` (append). v1 readers must tolerate the case where a file contains multiple "passes" — but in v1, no in-file `pass_id` exists, so concatenation is opaque. Force-append is for emergency / debug only.

## Crash semantics
- jsonl writes are line-buffered + per-step fsync.
- Sidecar writes are flushed + fsync'd on the same step-boundary cadence (so jsonl/sidecar advance together).
- Hard crash mid-step can leave the sidecar with a partial trailing record. Consistency scan (see validation) detects this and reports the trailing offset.

## Reader pseudocode

```python
import numpy as np
import struct

def open_sidecar(path):
    with open(path, "rb") as f:
        hdr = f.read(8)
    assert hdr[:4] == b'TKBN', f"bad magic: {hdr[:4]!r}"
    assert hdr[4] == 1, f"unsupported version {hdr[4]}"
    K = hdr[5]
    return K

def read_record(path, offset, K):
    with open(path, "rb") as f:
        f.seek(offset)
        step_id, layer_id, T = struct.unpack('<III', f.read(12))
        ids = np.frombuffer(f.read(T * K), dtype='<i1')          # shape (T*K,)
        w   = np.frombuffer(f.read(T * K * 2), dtype='<f2')      # shape (T*K,)
    return step_id, layer_id, T, ids, w
```

## Forward compatibility
- `version` byte at file offset 4 is the single source of truth for layout changes.
- v2 (future) could repurpose `reserved[6..7]` for a flags word (e.g. add `topk_T` per-record duplication, compressed payload, etc.).
- Readers MUST refuse to parse `version > known_max`.
