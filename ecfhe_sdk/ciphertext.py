"""FHECiphertext — serialisable wrapper around ecfhe.Ciphertext."""

from __future__ import annotations

import json
import struct
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import FHESession


def _pt_to_dict(P, params) -> dict:
    """BN254 G1 point → {"x": int, "y": int} in affine coords."""
    ax, ay = params.normalize(P)
    # py_ecc wraps integers in FQ objects; mcl uses attribute .get_str()
    def _int(v):
        if hasattr(v, "n"):
            return v.n
        if hasattr(v, "get_str"):
            return int(v.get_str(10))
        return int(v)
    return {"x": _int(ax), "y": _int(ay)}


def _pt_from_dict(d: dict, params):
    """{"x": int, "y": int} → BN254 G1 point."""
    x, y = d["x"], d["y"]
    # pyecc backend
    try:
        from py_ecc.fields.optimized_bn128_FQ import FQ
        return (FQ(x), FQ(y), FQ(1))
    except ImportError:
        pass
    # mcl backend
    import mcl as _mcl
    bn254 = getattr(_mcl, "bn254", None) or getattr(_mcl, "BN254", None)
    pt = bn254.G1()
    pt.set_str(f"1 {x} {y}")
    return pt


class FHECiphertext:
    """Opaque ciphertext with serialisation support.

    Do not construct directly — use ``FHESession.encrypt()`` or
    ``FHESession.ciphertext_from_bytes()``.
    """

    __slots__ = ("_ct", "_session")

    def __init__(self, raw_ct, session: "FHESession"):
        self._ct = raw_ct
        self._session = session

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        The dict contains the three G1 point components (C1, C2, C3) plus
        the precision parameter δ. The noise field is omitted — it is
        diagnostic metadata only and is not required for decryption.
        """
        from ecfhe import params as _p
        ct = self._ct
        return {
            "version": 1,
            "curve": "BN254",
            "C1": _pt_to_dict(ct.C1, _p),
            "C2": _pt_to_dict(ct.C2, _p),
            "C3": _pt_to_dict(ct.C3, _p),
            "delta": ct.delta,
            "mode": ct.mode,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def to_bytes(self) -> bytes:
        """Compact binary serialisation.

        Layout (all big-endian):
            4 bytes  magic   0x45434645  ('ECFE')
            1 byte   version 0x01
            32 bytes C1.x
            32 bytes C1.y
            32 bytes C2.x
            32 bytes C2.y
            32 bytes C3.x
            32 bytes C3.y
            4 bytes  delta (uint32)
        Total: 197 bytes
        """
        d = self.to_dict()
        parts = [b"ECFE", b"\x01"]
        for key in ("C1", "C2", "C3"):
            parts.append(d[key]["x"].to_bytes(32, "big"))
            parts.append(d[key]["y"].to_bytes(32, "big"))
        parts.append(struct.pack(">I", d["delta"]))
        return b"".join(parts)

    @classmethod
    def from_dict(cls, d: dict, session: "FHESession") -> "FHECiphertext":
        from ecfhe import params as _p
        from ecfhe.ciphertext import Ciphertext
        from ecfhe.noise import Noise

        C1 = _pt_from_dict(d["C1"], _p)
        C2 = _pt_from_dict(d["C2"], _p)
        C3 = _pt_from_dict(d["C3"], _p)
        raw = Ciphertext(C1=C1, C2=C2, C3=C3, delta=d["delta"],
                         noise=Noise(), mode=d.get("mode", "standard"))
        return cls(raw, session)

    @classmethod
    def from_json(cls, s: str, session: "FHESession") -> "FHECiphertext":
        return cls.from_dict(json.loads(s), session)

    @classmethod
    def from_bytes(cls, data: bytes, session: "FHESession") -> "FHECiphertext":
        if data[:4] != b"ECFE" or data[4] != 1:
            raise ValueError("invalid ecfhe ciphertext magic")
        offset = 5
        coords = []
        for _ in range(6):
            coords.append(int.from_bytes(data[offset:offset + 32], "big"))
            offset += 32
        (delta,) = struct.unpack(">I", data[offset:offset + 4])
        d = {
            "C1": {"x": coords[0], "y": coords[1]},
            "C2": {"x": coords[2], "y": coords[3]},
            "C3": {"x": coords[4], "y": coords[5]},
            "delta": delta,
        }
        return cls.from_dict(d, session)

    def __repr__(self) -> str:
        return f"<FHECiphertext delta={self._ct.delta}>"
