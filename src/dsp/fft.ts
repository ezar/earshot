/**
 * Minimal radix-2 FFT.
 *
 * The transform is allocation-free once an {@link Fft} instance exists, which
 * matters because the engine runs it on every 10 ms hop of every window.
 */

/** A reusable radix-2 complex FFT of a fixed size. */
export class Fft {
  /** Transform size, a power of two. */
  readonly size: number;

  private readonly cosTable: Float32Array;
  private readonly sinTable: Float32Array;
  private readonly reverse: Uint32Array;
  private readonly re: Float32Array;
  private readonly im: Float32Array;

  /**
   * @param size - Transform size in samples; must be a power of two.
   */
  constructor(size: number) {
    if (size < 2 || (size & (size - 1)) !== 0) {
      throw new Error(`Fft size must be a power of two, received ${size}`);
    }
    this.size = size;
    this.cosTable = new Float32Array(size / 2);
    this.sinTable = new Float32Array(size / 2);
    for (let i = 0; i < size / 2; i += 1) {
      const angle = (-2 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
    this.reverse = new Uint32Array(size);
    const bits = Math.log2(size);
    for (let i = 0; i < size; i += 1) {
      let r = 0;
      for (let b = 0; b < bits; b += 1) r |= ((i >>> b) & 1) << (bits - 1 - b);
      this.reverse[i] = r;
    }
    this.re = new Float32Array(size);
    this.im = new Float32Array(size);
  }

  /**
   * Transform a real signal and write its magnitude spectrum.
   *
   * @param input - Real samples; shorter inputs are zero-padded, longer ones truncated.
   * @param magnitude - Destination of length `size / 2 + 1`, filled with linear magnitudes.
   */
  realMagnitude(input: ArrayLike<number>, magnitude: Float32Array): void {
    const { size, re, im, reverse } = this;
    const n = Math.min(input.length, size);
    for (let i = 0; i < n; i += 1) {
      re[reverse[i] as number] = input[i] as number;
    }
    for (let i = n; i < size; i += 1) re[reverse[i] as number] = 0;
    im.fill(0);

    for (let half = 1; half < size; half <<= 1) {
      const step = size / (half << 1);
      for (let start = 0; start < size; start += half << 1) {
        for (let k = 0; k < half; k += 1) {
          const twiddle = k * step;
          const c = this.cosTable[twiddle] as number;
          const s = this.sinTable[twiddle] as number;
          const evenIndex = start + k;
          const oddIndex = evenIndex + half;
          const oddRe = re[oddIndex] as number;
          const oddIm = im[oddIndex] as number;
          const tr = oddRe * c - oddIm * s;
          const ti = oddRe * s + oddIm * c;
          re[oddIndex] = (re[evenIndex] as number) - tr;
          im[oddIndex] = (im[evenIndex] as number) - ti;
          re[evenIndex] = (re[evenIndex] as number) + tr;
          im[evenIndex] = (im[evenIndex] as number) + ti;
        }
      }
    }

    const bins = size / 2 + 1;
    for (let i = 0; i < bins; i += 1) {
      const r = re[i] as number;
      const q = im[i] as number;
      magnitude[i] = Math.sqrt(r * r + q * q);
    }
  }
}

/** Number of magnitude bins produced for a transform of `fftSize`. */
export function magnitudeBins(fftSize: number): number {
  return fftSize / 2 + 1;
}

/** Centre frequency of magnitude bin `index`, in Hz. */
export function binToHz(index: number, fftSize: number, sampleRateHz: number): number {
  return (index * sampleRateHz) / fftSize;
}

/** Build a periodic Hann window of `length` samples. */
export function hannWindow(length: number): Float32Array {
  const w = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / length));
  }
  return w;
}
