# lsb_dct.py
import cv2
import numpy as np

# ===============================
# LSB functions (operate on single-channel arrays)
# ===============================
def lsb_embed(channel_image, message_bits, start=0):
    """
    Embed list-of-bits (0/1) into the LSBs of a single-channel image (uint8).
    Returns a new uint8 image.
    """
    if channel_image is None:
        raise ValueError("channel_image is None")
    flat = channel_image.flatten().astype(np.uint8)
    bits = [int(b) for b in message_bits]

    if start + len(bits) > flat.size:
        raise ValueError("LSB capacity too small for this payload part.")

    mask = np.uint8(0xFE)  # 1111 1110
    for i, bit in enumerate(bits):
        idx = start + i
        flat[idx] = (flat[idx] & mask) | (bit & 1)

    return flat.reshape(channel_image.shape).astype(np.uint8)


def lsb_extract(channel_image, length, start=0):
    """
    Extract `length` bits from LSBs of single-channel image, returns list of ints (0/1).
    """
    flat = channel_image.flatten()
    if start >= flat.size:
        return []
    length = min(length, flat.size - start)
    return [int(flat[start + i] & 1) for i in range(length)]


# ===============================
# DCT functions (8x8 blocks; one bit per block using parity of a mid-band coef)
# ===============================
def dct_embed(gray_image, message_bits, block_size=8, start_block=0):
    """
    Embed bits using parity of a chosen DCT coefficient (one bit per 8x8 block).
    Operates on a single-channel (grayscale) image.
    """
    h, w = gray_image.shape
    embedded = gray_image.astype(np.float32).copy()
    blocks_y = h // block_size
    blocks_x = w // block_size
    total_blocks = blocks_y * blocks_x
    bits = [int(b) for b in message_bits]

    if start_block + len(bits) > total_blocks:
        raise ValueError("DCT capacity too small for this payload part.")

    bit_idx = 0
    # iterate block by block
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= len(bits):
                # done
                return np.clip(embedded, 0, 255).astype(np.uint8)

            sy, sx = by * block_size, bx * block_size
            block = embedded[sy:sy + block_size, sx:sx + block_size]
            dct_block = cv2.dct(block)
            # choose a mid-band coefficient (4,4) — safe for 8x8
            coeff = dct_block[4, 4]
            desired = bits[bit_idx]
            if (int(round(coeff)) & 1) != desired:
                # flip parity by nudging coefficient
                if coeff >= 0:
                    dct_block[4, 4] = coeff + 1.0
                else:
                    dct_block[4, 4] = coeff - 1.0
            block_recon = cv2.idct(dct_block)
            embedded[sy:sy + block_size, sx:sx + block_size] = block_recon
            bit_idx += 1

    return np.clip(embedded, 0, 255).astype(np.uint8)


def dct_extract(gray_image, length, block_size=8, start_block=0):
    """
    Extract bits from DCT parity (single-channel). Returns list of ints.
    """
    h, w = gray_image.shape
    blocks_y = h // block_size
    blocks_x = w // block_size
    total_blocks = blocks_y * blocks_x
    if start_block >= total_blocks:
        return []

    extracted = []
    bit_idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if bit_idx >= length:
                return extracted
            sy, sx = by * block_size, bx * block_size
            block = gray_image[sy:sy + block_size, sx:sx + block_size].astype(np.float32)
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue
            dct_block = cv2.dct(block)
            coef = dct_block[4, 4]
            extracted.append(int(round(coef)) & 1)
            bit_idx += 1

    return extracted


# ===============================
# HYBRID LSB + DCT (Blue LSB, Green DCT)
# ===============================
def hybrid_embed(image_bgr, message_bits):
    """
    Embed bits using:
      - Blue channel LSBs first (as many bits as Blue channel capacity allows)
      - Remaining bits (if any) in Green-channel DCT parity (one bit per 8x8 block)
    image_bgr : uint8 BGR image
    message_bits : list of ints (0/1)
    """
    if image_bgr is None:
        raise ValueError("image is None")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected BGR color image")

    b, g, r = cv2.split(image_bgr)
    b_capacity = b.size
    # DCT capacity = number of 8x8 blocks in green channel
    dct_capacity = (g.shape[0] // 8) * (g.shape[1] // 8)

    total_capacity = b_capacity + dct_capacity
    if len(message_bits) > total_capacity:
        raise ValueError(f"Payload too large for image. Need {len(message_bits)} bits, capacity {total_capacity}.")

    # embed into Blue LSB first
    if len(message_bits) <= b_capacity:
        b_stego = lsb_embed(b.copy(), message_bits, start=0)
        g_stego = g.copy()
    else:
        bits_b = message_bits[:b_capacity]
        bits_g = message_bits[b_capacity:]
        b_stego = lsb_embed(b.copy(), bits_b, start=0)
        g_stego = dct_embed(g.copy(), bits_g, block_size=8, start_block=0)

    stego = cv2.merge([b_stego, g_stego, r])
    return stego.astype(np.uint8)


def hybrid_extract(image_bgr, total_length):
    """
    Extract first `total_length` bits using the hybrid layout described above.
    Returns list of ints (0/1).
    """
    if image_bgr is None:
        return []

    b, g, r = cv2.split(image_bgr)
    b_capacity = b.size

    if total_length <= b_capacity:
        return lsb_extract(b, total_length, start=0)
    else:
        bits_b = lsb_extract(b, b_capacity, start=0)
        remaining = total_length - b_capacity
        bits_g = dct_extract(g, remaining, block_size=8, start_block=0)
        return bits_b + bits_g


# ===============================
# Helpers: convert bytes <-> bits
# ===============================
def text_to_bits(data: bytes):
    """
    Convert bytes to list of bits (MSB-first per byte).
    """
    return [int(bit) for byte in data for bit in format(byte, '08b')]


def bits_to_text(bits):
    """
    Convert list of bits to a UTF-8 string (ignore incomplete trailing byte).
    """
    byte_values = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            break
        byte_values.append(int(''.join(map(str, chunk)), 2))
    return bytes(byte_values).decode('utf-8', errors='ignore')
