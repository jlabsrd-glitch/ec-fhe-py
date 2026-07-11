"""Default backend: py_ecc.optimized_bn128 (pure Python, slow but
portable)."""

from py_ecc import optimized_bn128 as _bn

G1 = _bn.G1
G2 = _bn.G2
Z1 = _bn.Z1
Z2 = _bn.Z2

ORDER: int = _bn.curve_order
FIELD_MOD: int = _bn.field_modulus

FQ = _bn.FQ
FQ2 = _bn.FQ2
FQ12 = _bn.FQ12

add = _bn.add
multiply = _bn.multiply
neg = _bn.neg
pairing = _bn.pairing
normalize = _bn.normalize
is_inf = _bn.is_inf
