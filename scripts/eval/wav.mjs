/**
 * A small WAV reader for the evaluation harness.
 *
 * The datasets ship 16-bit PCM WAV at 16 kHz (MIMII, ToyADMOS) or 8 kHz
 * (CatMeows), so only uncompressed PCM is supported. Anything else should be
 * converted before evaluation rather than silently mis-decoded.
 */

/**
 * Decode a WAV buffer to mono Float32 at 16 kHz.
 *
 * @param {Buffer} buffer - Raw file contents.
 * @returns {Float32Array} Mono samples in [-1, 1] at 16 kHz.
 */
export function decodeWavToMono16k(buffer) {
  const { samples, sampleRate, channels } = decodeWav(buffer);
  const mono = channels === 1 ? samples : mixToMono(samples, channels);
  return sampleRate === 16000 ? mono : resampleLinear(mono, sampleRate, 16000);
}

/** Parse a PCM WAV file into interleaved Float32 samples. */
export function decodeWav(buffer) {
  if (buffer.length < 12 || buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('eval: not a RIFF/WAVE file');
  }
  let offset = 12;
  let format = null;
  let data = null;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const body = offset + 8;
    if (id === 'fmt ') {
      format = {
        audioFormat: buffer.readUInt16LE(body),
        channels: buffer.readUInt16LE(body + 2),
        sampleRate: buffer.readUInt32LE(body + 4),
        bitsPerSample: buffer.readUInt16LE(body + 14),
      };
    } else if (id === 'data') {
      data = buffer.subarray(body, Math.min(body + size, buffer.length));
    }
    offset = body + size + (size % 2);
  }
  if (format === null || data === null) throw new Error('eval: WAV is missing a fmt or data chunk');
  if (format.audioFormat !== 1 && format.audioFormat !== 3) {
    throw new Error(`eval: unsupported WAV format ${format.audioFormat}; convert to PCM first`);
  }

  const { bitsPerSample } = format;
  const bytes = bitsPerSample / 8;
  const count = Math.floor(data.length / bytes);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    const at = i * bytes;
    if (format.audioFormat === 3 && bitsPerSample === 32) samples[i] = data.readFloatLE(at);
    else if (bitsPerSample === 16) samples[i] = data.readInt16LE(at) / 32768;
    else if (bitsPerSample === 24) samples[i] = (data.readIntLE(at, 3) ?? 0) / 8388608;
    else if (bitsPerSample === 32) samples[i] = data.readInt32LE(at) / 2147483648;
    else if (bitsPerSample === 8) samples[i] = (data.readUInt8(at) - 128) / 128;
    else throw new Error(`eval: unsupported bit depth ${bitsPerSample}`);
  }
  return { samples, sampleRate: format.sampleRate, channels: format.channels };
}

function mixToMono(interleaved, channels) {
  const frames = Math.floor(interleaved.length / channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i += 1) {
    let sum = 0;
    for (let c = 0; c < channels; c += 1) sum += interleaved[i * channels + c];
    out[i] = sum / channels;
  }
  return out;
}

/**
 * Linear resampling.
 *
 * Good enough for the evaluation harness because every dataset is at or below
 * 16 kHz, so no aliasing is introduced. Runtime audio goes through
 * `src/dsp/resample.ts`, which low-passes first.
 */
function resampleLinear(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.max(1, Math.floor(input.length / ratio)));
  for (let i = 0; i < out.length; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const a = input[index] ?? 0;
    const b = input[index + 1] ?? a;
    out[i] = a + (b - a) * (position - index);
  }
  return out;
}
