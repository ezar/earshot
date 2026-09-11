/**
 * Optional TensorFlow.js classification head, published as the `earshot/mlp`
 * entry point.
 *
 * kNN is the right default: no training step, useful from one example, and
 * every prediction is explainable. Once a user has hundreds of examples per
 * label a small MLP over the frozen embedding does better, so earshot offers
 * one — behind its own entry point, so the core bundle never pulls TensorFlow.
 *
 * ```ts
 * import { createMlpHead } from 'earshot/mlp';
 * const head = await createMlpHead({ inputDimensions: 1024, labels: ['luna', 'sol'] });
 * await head.train(examples);
 * const prediction = await head.predict(embedding);
 * ```
 */

import { l2Normalize } from '../util/math.js';
import type { Example, LabelScore, Prediction } from './knn.js';

/** The slice of TensorFlow.js earshot uses. */
export interface TfjsModule {
  sequential(config?: unknown): TfjsModel;
  tensor2d(values: number[][], shape?: [number, number]): TfjsTensor;
  readonly layers: {
    dense(config: Record<string, unknown>): unknown;
    dropout(config: Record<string, unknown>): unknown;
  };
  readonly train: { adam(learningRate?: number): unknown };
}

/** A TensorFlow.js sequential model. */
export interface TfjsModel {
  add(layer: unknown): void;
  compile(config: Record<string, unknown>): void;
  fit(x: TfjsTensor, y: TfjsTensor, config?: Record<string, unknown>): Promise<{ history: Record<string, number[]> }>;
  predict(x: TfjsTensor): TfjsTensor;
  dispose(): void;
}

/** A TensorFlow.js tensor. */
export interface TfjsTensor {
  data(): Promise<Float32Array>;
  dispose(): void;
}

/** Options for {@link createMlpHead}. */
export interface MlpOptions<Label extends string = string> {
  /** Dimensionality of the input embeddings, e.g. 1024 for YAMNet. */
  readonly inputDimensions: number;
  /** Labels the head can predict, in output order. */
  readonly labels: readonly Label[];
  /** Units in the hidden layer. Defaults to 64. */
  readonly hiddenUnits?: number;
  /** Dropout applied after the hidden layer, in `[0, 1)`. Defaults to 0.3. */
  readonly dropout?: number;
  /** Adam learning rate. Defaults to 0.002. */
  readonly learningRate?: number;
  /** Override the module loader, e.g. to bundle TensorFlow.js statically. */
  readonly loadTfjs?: () => Promise<TfjsModule>;
}

/** Options for {@link MlpHead.train}. */
export interface TrainOptions {
  /** Passes over the training set. Defaults to 40. */
  readonly epochs?: number;
  /** Examples per gradient step. Defaults to 16. */
  readonly batchSize?: number;
  /** Fraction held out for validation, in `[0, 1)`. Defaults to 0.2. */
  readonly validationSplit?: number;
  /** Called after every epoch with its index and loss. */
  readonly onEpoch?: (epoch: number, loss: number) => void;
}

/** The outcome of training. */
export interface TrainResult {
  /** Final training loss. */
  readonly loss: number;
  /** Final validation accuracy, or null when nothing was held out. */
  readonly validationAccuracy: number | null;
  /** Epochs actually run. */
  readonly epochs: number;
}

/** A trainable classification head over frozen embeddings. */
export interface MlpHead<Label extends string = string> {
  /**
   * Train on labelled examples.
   *
   * @param examples - Examples whose labels are all in `options.labels`.
   * @throws When an example carries an unknown label.
   */
  train(examples: readonly Example<Label>[], options?: TrainOptions): Promise<TrainResult>;
  /** Classify one embedding; the shape matches {@link KnnClassifier.predict}. */
  predict(embedding: ArrayLike<number>): Promise<Prediction<Label>>;
  /** Release the TensorFlow.js graph. */
  dispose(): void;
  /** The labels in output order. */
  readonly labels: readonly Label[];
}

const defaultLoadTfjs = async (): Promise<TfjsModule> => {
  const specifier = '@tensorflow/tfjs';
  const module: unknown = await import(/* @vite-ignore */ specifier);
  return module as TfjsModule;
};

/**
 * Build an untrained MLP head.
 *
 * @param options - Input shape, label set and architecture.
 * @throws When fewer than two labels are supplied.
 */
export async function createMlpHead<Label extends string = string>(
  options: MlpOptions<Label>,
): Promise<MlpHead<Label>> {
  if (options.labels.length < 2) {
    throw new Error('earshot: createMlpHead needs at least two labels');
  }
  const tf = await (options.loadTfjs ?? defaultLoadTfjs)();
  const labels = options.labels.slice();

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: options.hiddenUnits ?? 64,
      activation: 'relu',
      inputShape: [options.inputDimensions],
    }),
  );
  model.add(tf.layers.dropout({ rate: options.dropout ?? 0.3 }));
  model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(options.learningRate ?? 0.002),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return {
    labels,
    async train(examples, trainOptions = {}): Promise<TrainResult> {
      if (examples.length === 0) throw new Error('earshot: MlpHead.train needs at least one example');
      const x: number[][] = [];
      const y: number[][] = [];
      for (const example of examples) {
        const index = labels.indexOf(example.label);
        if (index < 0) throw new Error(`earshot: example carries unknown label "${example.label}"`);
        x.push(Array.from(l2Normalize(example.embedding)));
        const oneHot = new Array<number>(labels.length).fill(0);
        oneHot[index] = 1;
        y.push(oneHot);
      }

      const epochs = trainOptions.epochs ?? 40;
      const xs = tf.tensor2d(x);
      const ys = tf.tensor2d(y);
      try {
        const history = await model.fit(xs, ys, {
          epochs,
          batchSize: trainOptions.batchSize ?? 16,
          validationSplit: trainOptions.validationSplit ?? 0.2,
          shuffle: true,
          callbacks:
            trainOptions.onEpoch === undefined
              ? undefined
              : {
                  onEpochEnd: (epoch: number, logs?: Record<string, number>) => {
                    trainOptions.onEpoch?.(epoch, logs?.['loss'] ?? 0);
                  },
                },
        });
        const losses = history.history['loss'] ?? [];
        const accuracies = history.history['val_acc'] ?? history.history['val_accuracy'] ?? [];
        return {
          loss: losses[losses.length - 1] ?? 0,
          validationAccuracy: accuracies.length === 0 ? null : (accuracies[accuracies.length - 1] as number),
          epochs,
        };
      } finally {
        xs.dispose();
        ys.dispose();
      }
    },
    async predict(embedding): Promise<Prediction<Label>> {
      const input = tf.tensor2d([Array.from(l2Normalize(embedding))]);
      const output = model.predict(input);
      try {
        const probabilities = await output.data();
        const scores: LabelScore<Label>[] = labels
          .map((label, i) => ({ label, score: probabilities[i] ?? 0 }))
          .sort((a, b) => b.score - a.score);
        const best = scores[0];
        return {
          label: best?.label ?? null,
          confidence: best?.score ?? 0,
          scores,
          neighbours: [],
        };
      } finally {
        input.dispose();
        output.dispose();
      }
    },
    dispose(): void {
      model.dispose();
    },
  };
}
