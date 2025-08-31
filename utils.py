# utils.py
import cv2
import numpy as np
import hashlib
from lsb_dct import hybrid_embed, hybrid_extract, text_to_bits, bits_to_text

# ------------ helpers ------------
def _len_prefix_bits(n_bytes: int) -> list[int]:
    """32-bit big-endian length (bytes) as list of bits."""
    return [int(b) for b in f"{n_bytes:032b}"]


def _apply_key_wrap(message: str, key: str | None) -> str:
    if not key:
        return message
    h = hashlib.sha256(str(key).encode()).hexdigest()
    return f"{h[:16]}{message}{h[-16:]}"


def _remove_key_wrap(wrapped: str, key: str | None) -> str:
    if not key:
        return wrapped
    h = hashlib.sha256(str(key).encode()).hexdigest()
    if wrapped.startswith(h[:16]) and wrapped.endswith(h[-16:]):
        return wrapped[16:-16]
    return "Invalid Key!"


# ------------ embedding / extraction using HYBRID LSB + DCT ------------
def embed_message(image_bgr: np.ndarray,
                  message: str,
                  key: str | None = None) -> np.ndarray:
    """
    Prepare payload (header+message) as bits and call hybrid_embed.
    Header is a 32-bit big-endian length (bytes).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image array.")

    wrapped = _apply_key_wrap(message, key)
    payload_bytes = wrapped.encode("utf-8")

    header_bits = _len_prefix_bits(len(payload_bytes))  # 32 bits
    payload_bits = text_to_bits(payload_bytes)         # list of ints from bytes
    all_bits = header_bits + payload_bits

    # capacity = Blue LSB capacity + Green DCT capacity
    h, w = image_bgr.shape[:2]
    b_capacity = h * w  # blue channel LSB capacity (one bit per pixel element)
    dct_capacity = (h // 8) * (w // 8)  # one bit per 8x8 block in green channel
    total_capacity = b_capacity + dct_capacity

    if len(all_bits) > total_capacity:
        raise ValueError(f"Message too large! Need {len(all_bits)} bits, capacity {total_capacity}.")

    stego_bgr = hybrid_embed(image_bgr.copy(), all_bits)
    return stego_bgr


def extract_message(stego_bgr: np.ndarray, key: str | None = None) -> str:
    """
    Read 32-bit header first, compute payload length (bytes), then extract exact bit count.
    """
    if stego_bgr is None or stego_bgr.size == 0:
        raise ValueError("Invalid image array.")

    header_bits = hybrid_extract(stego_bgr, 32)
    if len(header_bits) < 32:
        raise ValueError("Failed to read header bits from image.")

    msg_len_bytes = int(''.join(map(str, header_bits)), 2)
    total_bits = 32 + msg_len_bytes * 8

    all_bits = hybrid_extract(stego_bgr, total_bits)
    if len(all_bits) < total_bits:
        # pad with zeros if truncated (shouldn't happen when capacities were checked)
        all_bits += [0] * (total_bits - len(all_bits))

    payload_bits = all_bits[32:]
    recovered = bits_to_text(payload_bits)
    return _remove_key_wrap(recovered, key)
