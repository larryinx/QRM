"""QRM Model Module.

This module provides the QRM (Quantized Recursive Model) implementation:
- QRMConfig: Configuration class extending TRMConfig
- QRMInner: Core model with FSQ integration
- QRMForPuzzleSolving: Full model with loss computation
"""

from qrm.models.qrm.configuration_qrm import QRMConfig
from qrm.models.qrm.modeling_qrm import (
    QRMCarry,
    QRMForPuzzleSolving,
    QRMInner,
    QRMInnerCarry,
    QRMOutput,
)

__all__ = [
    "QRMConfig",
    "QRMInner",
    "QRMInnerCarry",
    "QRMCarry",
    "QRMForPuzzleSolving",
    "QRMOutput",
]
