/**
 * Microphone capture.
 *
 * earshot deliberately turns off every browser voice-processing feature:
 * echo cancellation, noise suppression and automatic gain control all rewrite
 * the signal in ways that destroy the stationarity a machine profile depends
 * on, and AGC in particular makes dBFS levels incomparable between checks.
 */

/// <reference lib="dom" />

import { SAMPLE_RATE_HZ } from '../constants.js';
import { resample } from '../dsp/resample.js';
import { CAPTURE_PROCESSOR_NAME, type CaptureChunkMessage } from './capture-worklet.js';

/** What the browser actually granted, compared with what earshot asked for. */
export interface AppliedConstraints {
  /** Hardware sample rate of the capture graph, in Hz. */
  readonly sampleRateHz: number;
  /** True when the browser confirms echo cancellation is off. */
  readonly echoCancellation: boolean;
  /** True when the browser confirms noise suppression is off. */
  readonly noiseSuppression: boolean;
  /** True when the browser confirms automatic gain control is off. */
  readonly autoGainControl: boolean;
  /** Device label, when the permission grant exposes it. */
  readonly deviceLabel: string;
  /**
   * Names of the constraints earshot requested but the browser did not honour.
   * A non-empty list means levels may not be comparable across checks.
   */
  readonly unhonoured: readonly string[];
}

/** Options for {@link createCapture}. */
export interface CaptureOptions {
  /** URL of the `earshot/capture-worklet` module, loaded with `?url`. */
  readonly workletUrl: string;
  /** `deviceId` of the input to open; defaults to the system default input. */
  readonly deviceId?: string;
  /** Samples per chunk posted from the worklet. Defaults to 2048. */
  readonly chunkSamples?: number;
  /** Resample chunks to this rate before delivering them, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly targetSampleRateHz?: number;
  /** Existing AudioContext to reuse; one is created when omitted. */
  readonly audioContext?: AudioContext;
}

/** A running microphone capture. */
export interface Capture {
  /** What the browser actually applied. */
  readonly appliedConstraints: AppliedConstraints;
  /** Sample rate of the buffers delivered to subscribers, in Hz. */
  readonly sampleRateHz: number;
  /**
   * Subscribe to resampled mono chunks.
   *
   * @returns An unsubscribe function.
   */
  onChunk(listener: (samples: Float32Array) => void): () => void;
  /** Async iterator over the same chunks, for `for await` consumers. */
  chunks(): AsyncIterableIterator<Float32Array>;
  /** Stop capture, release the microphone and close the context earshot created. */
  stop(): Promise<void>;
}

/**
 * Open the microphone and start an AudioWorklet capture.
 *
 * @param options - Worklet URL and optional device selection.
 * @returns A {@link Capture} delivering mono chunks at `targetSampleRateHz`.
 * @throws When `getUserMedia` is unavailable or the user denies permission.
 */
export async function createCapture(options: CaptureOptions): Promise<Capture> {
  const targetSampleRateHz = options.targetSampleRateHz ?? SAMPLE_RATE_HZ;
  if (typeof navigator === 'undefined' || navigator.mediaDevices?.getUserMedia === undefined) {
    throw new Error('earshot: getUserMedia is not available in this environment');
  }

  const audioConstraints: MediaTrackConstraints = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
    ...(options.deviceId === undefined ? {} : { deviceId: { exact: options.deviceId } }),
  };
  const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints, video: false });

  const ownsContext = options.audioContext === undefined;
  const context = options.audioContext ?? new AudioContext();
  if (context.state === 'suspended') await context.resume();

  try {
    await context.audioWorklet.addModule(options.workletUrl);
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    if (ownsContext) await context.close();
    throw new Error(`earshot: failed to load the capture worklet from ${options.workletUrl}`, { cause: error });
  }

  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, CAPTURE_PROCESSOR_NAME, {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    channelCount: 1,
    channelCountMode: 'explicit',
    processorOptions: { chunkSamples: options.chunkSamples ?? 2048 },
  });
  source.connect(node);

  const appliedConstraints = readAppliedConstraints(stream, context.sampleRate);
  const listeners = new Set<(samples: Float32Array) => void>();

  node.port.onmessage = (event: MessageEvent) => {
    const message = event.data as CaptureChunkMessage | { type: string };
    if (message.type !== 'chunk') return;
    const chunk = (message as CaptureChunkMessage).samples;
    const samples =
      context.sampleRate === targetSampleRateHz ? chunk : resample(chunk, context.sampleRate, targetSampleRateHz);
    for (const listener of listeners) listener(samples);
  };

  let stopped = false;
  const capture: Capture = {
    appliedConstraints,
    sampleRateHz: targetSampleRateHz,
    onChunk(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    chunks(): AsyncIterableIterator<Float32Array> {
      return createChunkIterator((listener) => capture.onChunk(listener), () => stopped);
    },
    async stop(): Promise<void> {
      if (stopped) return;
      stopped = true;
      node.port.onmessage = null;
      listeners.clear();
      source.disconnect();
      node.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      if (ownsContext) await context.close();
    },
  };
  return capture;
}

/** Read back the constraints the browser actually applied to the audio track. */
function readAppliedConstraints(stream: MediaStream, sampleRateHz: number): AppliedConstraints {
  const track = stream.getAudioTracks()[0];
  const settings = (track?.getSettings() ?? {}) as MediaTrackSettings;
  const unhonoured: string[] = [];
  const check = (name: 'echoCancellation' | 'noiseSuppression' | 'autoGainControl'): boolean => {
    const value = settings[name];
    // `undefined` means the browser does not report the setting; treat that as
    // honoured but record it so the app can warn if it wants to.
    if (value === undefined) return true;
    if (value) unhonoured.push(name);
    return !value;
  };
  return {
    sampleRateHz: settings.sampleRate ?? sampleRateHz,
    echoCancellation: check('echoCancellation'),
    noiseSuppression: check('noiseSuppression'),
    autoGainControl: check('autoGainControl'),
    deviceLabel: track?.label ?? '',
    unhonoured,
  };
}

/** Bridge a push-based listener to a pull-based async iterator with a bounded queue. */
function createChunkIterator(
  subscribe: (listener: (samples: Float32Array) => void) => () => void,
  isStopped: () => boolean,
): AsyncIterableIterator<Float32Array> {
  const queue: Float32Array[] = [];
  let pending: ((result: IteratorResult<Float32Array>) => void) | null = null;
  const unsubscribe = subscribe((samples) => {
    if (pending !== null) {
      const resolve = pending;
      pending = null;
      resolve({ value: samples, done: false });
      return;
    }
    // Drop the oldest chunk rather than grow without bound if the consumer stalls.
    if (queue.length > 64) queue.shift();
    queue.push(samples);
  });

  return {
    [Symbol.asyncIterator]() {
      return this;
    },
    async next(): Promise<IteratorResult<Float32Array>> {
      const queued = queue.shift();
      if (queued !== undefined) return { value: queued, done: false };
      if (isStopped()) return { value: undefined, done: true };
      return new Promise<IteratorResult<Float32Array>>((resolve) => {
        pending = resolve;
      });
    },
    async return(): Promise<IteratorResult<Float32Array>> {
      unsubscribe();
      return { value: undefined, done: true };
    },
  };
}
