"""
torus-ecfhe — EC-ElGamal Fully Homomorphic Encryption SDK
BN254 curve · additive HE + Weil pairing multiplication

Quick start:
    from ecfhe_sdk import FHESession

    s = FHESession(max_value=10_000)      # max plaintext value for BSGS table
    a = s.encrypt(42)
    b = s.encrypt(58)

    print(s.decrypt(s.add(a, b)))         # 100
    print(s.decrypt(s.mul_scalar(a, 3)))  # 126
    print(s.decrypt(s.mul(a, b)))         # 2436  (via Weil pairing)

    # Serialise a ciphertext (share between parties)
    data = a.to_bytes()
    a2   = s.ciphertext_from_bytes(data)
    print(s.decrypt(a2))                  # 42
"""

from .session import FHESession
from .ciphertext import FHECiphertext

__all__ = ["FHESession", "FHECiphertext"]
__version__ = "1.0.0"
