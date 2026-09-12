/**
 * Minimal radix-2 FFT.
 *
 * The transform is allocation-free once an {@link Fft} instance exists, which
 * matters because the engine runs it on every 10 ms hop of every window.
 */

/**
 * A reusable radix-2 FFT specialised for real input.
 *
 * A real signal of length `N` is transformed with a complex FFT of length `N/2`
 * plus a recombination pass, rather than a full `N`-point complex transform with
 * a zeroed imaginary part. That halves the butterfly work, which matters because
 * the engine runs 96 of these per analysis window.
 */
export class Fft {
  /** Transform size, a power of two. */
  readonly size: number;

  /** Half the transform size: the length of the internal complex FFT. */
  private readonly half: number;

  private readonly cosTable: Float64Array;
  private readonly sinTable: Float64Array;
  private readonly reverse: Uint32Array;
  /** Recombination twiddles, `e^(-2*pi*i*k/size)` for `k` in `[0, half]`. */
  private readonly recombineCos: Float64Array;
  private readonly recombineSin: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;

  /**
   * @param size - Transform size in samples; must be a power of two and at least 4.
   */
  constructor(size: number) {
    if (size < 4 || (size & (size - 1)) !== 0) {
      throw new Error(`Fft size must be a power of two of at least 4, received ${size}`);
    }
    this.size = size;
    const half = size / 2;
    this.half = half;

    this.cosTable = new Float64Array(half / 2);
    this.sinTable = new Float64Array(half / 2);
    for (let i = 0; i < half / 2; i += 1) {
      const angle = (-2 * Math.PI * i) / half;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }

    this.reverse = new Uint32Array(half);
    const bits = Math.log2(half);
    for (let i = 0; i < half; i += 1) {
      let r = 0;
      for (let b = 0; b < bits; b += 1) r |= ((i >>> b) & 1) << (bits - 1 - b);
      this.reverse[i] = r;
    }

    this.recombineCos = new Float64Array(half + 1);
    this.recombineSin = new Float64Array(half + 1);
    for (let k = 0; k <= half; k += 1) {
      const angle = (-2 * Math.PI * k) / size;
      this.recombineCos[k] = Math.cos(angle);
      this.recombineSin[k] = Math.sin(angle);
    }

    this.re = new Float64Array(half);
    this.im = new Float64Array(half);
  }

  /**
   * Transform a real signal and write its magnitude spectrum.
   *
   * @param input - Real samples; shorter inputs are zero-padded, longer ones truncated.
   * @param magnitude - Destination of length `size / 2 + 1`, filled with linear magnitudes.
   */
  realMagnitude(input: ArrayLike<number>, magnitude: Float32Array): void {
    const { size, half, re, im, reverse } = this;
    const n = Math.min(input.length, size);

    // Pack the real signal into half as many complex samples: the even-indexed
    // samples become the real parts and the odd-indexed ones the imaginary
    // parts. Bit-reversal is applied here so the butterflies below can run in
    // place without a separate permutation pass.
    for (let i = 0; i < half; i += 1) {
      const target = reverse[i] as number;
      const even = 2 * i;
      const odd = even + 1;
      re[target] = even < n ? (input[even] as number) : 0;
      im[target] = odd < n ? (input[odd] as number) : 0;
    }

    for (let span = 1; span < half; span <<= 1) {
      const step = half / (span << 1);
      for (let start = 0; start < half; start += span << 1) {
        for (let k = 0; k < span; k += 1) {
          const twiddle = k * step;
          const c = this.cosTable[twiddle] as number;
          const s = this.sinTable[twiddle] as number;
          const even = start + k;
          const odd = even + span;
          const oddRe = re[odd] as number;
          const oddIm = im[odd] as number;
          const tr = oddRe * c - oddIm * s;
          const ti = oddRe * s + oddIm * c;
          re[odd] = (re[even] as number) - tr;
          im[odd] = (im[even] as number) - ti;
          re[even] = (re[even] as number) + tr;
          im[even] = (im[even] as number) + ti;
        }
      }
    }

    // Recombine the half-length complex spectrum into the real spectrum. The
    // even-indexed and odd-indexed subsequences are recovered from the
    // Hermitian symmetry of the packed transform, then combined with a twiddle.
    const zeroRe = re[0] as number;
    const zeroIm = im[0] as number;
    magnitude[0] = Math.abs(zeroRe + zeroIm);
    magnitude[half] = Math.abs(zeroRe - zeroIm);

    for (let k = 1; k < half; k += 1) {
      const mirror = half - k;
      const kr = re[k] as number;
      const ki = im[k] as number;
      const mr = re[mirror] as number;
      const mi = im[mirror] as number;

      const evenRe = 0.5 * (kr + mr);
      const evenIm = 0.5 * (ki - mi);
      const oddRe = 0.5 * (ki + mi);
      const oddIm = -0.5 * (kr - mr);

      const wr = this.recombineCos[k] as number;
      const wi = this.recombineSin[k] as number;
      const realPart = evenRe + (oddRe * wr - oddIm * wi);
      const imagPart = evenIm + (oddRe * wi + oddIm * wr);
      magnitude[k] = Math.sqrt(realPart * realPart + imagPart * imagPart);
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
