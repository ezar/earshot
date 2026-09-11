/**
 * Embedding quantization.
 *
 * A YAMNet embedding is 1024 float32 values, 4 KiB each. A user with a few
 * hundred stored examples is carrying megabytes in IndexedDB for no benefit:
 * float16 halves that with no measurable effect on cosine distance, and int8
 * quarters it at a cost of roughly 0.002 in cosine similarity on normalized
 * vectors.
 */

import type { QuantizedEmbedding } from './types.js';

/** Quantization schemes {@link quantize} supports. */
export type QuantizationFormat = 'float16' | 'int8';

/**
 * Quantize an embedding into a persistable object.
 *
 * @param embedding - The vector to compress.
 * @param format - `'float16'` (default) or `'int8'`.
 */
export function quantize(embedding: ArrayLike<number>, format: QuantizationFormat = 'float16'): QuantizedEmbedding {
  return format === 'int8' ? quantizeInt8(embedding) : quantizeFloat16(embedding);
}

/**
 * Restore an embedding quantized by {@link quantize}.
 *
 * @throws When the payload length does not match the declared dimensions.
 */
export function dequantize(quantized: QuantizedEmbedding): Float32Array {
  const bytes = base64ToBytes(quantized.data);
  if (quantized.format === 'int8') {
    if (bytes.length !== quantized.dimensions) {
      throw new Error(`earshot: int8 payload has ${bytes.length} bytes, expected ${quantized.dimensions}`);
    }
    const out = new Float32Array(quantized.dimensions);
    const signed = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length);
    for (let i = 0; i < out.length; i += 1) out[i] = (signed[i] as number) * quantized.scale;
    return out;
  }
  if (bytes.length !== quantized.dimensions * 2) {
    throw new Error(`earshot: float16 payload has ${bytes.length} bytes, expected ${quantized.dimensions * 2}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const out = new Float32Array(quantized.dimensions);
  for (let i = 0; i < out.length; i += 1) out[i] = float16ToFloat32(view.getUint16(i * 2, true));
  return out;
}

function quantizeFloat16(embedding: ArrayLike<number>): QuantizedEmbedding {
  const bytes = new Uint8Array(embedding.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < embedding.length; i += 1) {
    view.setUint16(i * 2, float32ToFloat16(embedding[i] as number), true);
  }
  return {
    format: 'float16',
    dimensions: embedding.length,
    data: bytesToBase64(bytes),
    scale: 1,
  };
}

function quantizeInt8(embedding: ArrayLike<number>): QuantizedEmbedding {
  let peak = 0;
  for (let i = 0; i < embedding.length; i += 1) {
    const magnitude = Math.abs(embedding[i] as number);
    if (magnitude > peak) peak = magnitude;
  }
  const scale = peak === 0 ? 1 : peak / 127;
  const signed = new Int8Array(embedding.length);
  for (let i = 0; i < embedding.length; i += 1) {
    const value = Math.round((embedding[i] as number) / scale);
    signed[i] = Math.max(-127, Math.min(127, value));
  }
  return {
    format: 'int8',
    dimensions: embedding.length,
    data: bytesToBase64(new Uint8Array(signed.buffer)),
    scale,
  };
}

/**
 * Convert a float32 to the IEEE 754 binary16 bit pattern.
 *
 * Values outside the half-precision range saturate to infinity, and
 * subnormals are preserved; earshot's embeddings live well inside the range,
 * so neither path is normally taken.
 */
export function float32ToFloat16(value: number): number {
  const buffer = FLOAT_CONVERSION_VIEW;
  buffer.setFloat32(0, value, true);
  const bits = buffer.getUint32(0, true);
  const sign = (bits >>> 16) & 0x8000;
  let exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;

  if (exponent === 0xff) {
    // Infinity or NaN.
    return sign | 0x7c00 | (mantissa === 0 ? 0 : 0x0200);
  }
  exponent = exponent - 127 + 15;
  if (exponent >= 0x1f) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exponent;
    const rounded = (mantissa + (1 << (shift - 1))) >>> shift;
    return sign | rounded;
  }
  const rounded = (mantissa + 0x1000) >>> 13;
  if (rounded & 0x400) return sign | ((exponent + 1) << 10);
  return sign | (exponent << 10) | rounded;
}

/** Convert an IEEE 754 binary16 bit pattern back to a float32 value. */
export function float16ToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

const FLOAT_CONVERSION_VIEW = new DataView(new ArrayBuffer(4));

/** Base64-encode a byte array without depending on Node or DOM specifics. */
export function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return globalThis.btoa(binary);
}

/** Decode a base64 string produced by {@link bytesToBase64}. */
export function base64ToBytes(text: string): Uint8Array {
  const binary = globalThis.atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
