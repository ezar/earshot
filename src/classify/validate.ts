/**
 * Stratified k-fold self-test.
 *
 * With ten examples per cat a user has no held-out set, so Meowlogue tells them
 * how well the classifier does on its own examples by repeatedly hiding a fold
 * and predicting it. The number is honest about small samples: a label with one
 * example is reported, not silently dropped.
 */

import { createRandom } from '../util/math.js';
import { predictWith, type Example } from './knn.js';

/** Per-label accuracy. */
export interface LabelReport<Label extends string = string> {
  readonly label: Label;
  /** Examples carrying this label. */
  readonly support: number;
  /** Fraction of them predicted correctly, in `[0, 1]`. */
  readonly recall: number;
  /** Fraction of predictions of this label that were right, in `[0, 1]`. */
  readonly precision: number;
}

/** The outcome of a cross-validation run. */
export interface ValidationReport<Label extends string = string> {
  /** Overall fraction correct, in `[0, 1]`. */
  readonly accuracy: number;
  /** Folds actually run. */
  readonly folds: number;
  /** Examples evaluated. */
  readonly total: number;
  /** Per-label breakdown, most-supported first. */
  readonly labels: readonly LabelReport<Label>[];
  /** `confusion[trueLabel][predictedLabel]` counts; `""` means no prediction. */
  readonly confusion: Readonly<Record<string, Readonly<Record<string, number>>>>;
}

/** Options for {@link crossValidate}. */
export interface ValidateOptions {
  /** Folds to run. Defaults to 5, capped at the smallest label's support. */
  readonly folds?: number;
  /** Neighbours per prediction. Defaults to 5. */
  readonly k?: number;
  /** Seed for the deterministic shuffle. Defaults to 1. */
  readonly seed?: number;
}

/**
 * Run a stratified k-fold self-test over a set of labelled examples.
 *
 * @param examples - Every stored example; at least two labels are needed.
 * @param options - Fold count, neighbour count and shuffle seed.
 * @returns Accuracy, per-label recall and precision, and a confusion matrix.
 */
export function crossValidate<Label extends string = string>(
  examples: readonly Example<Label>[],
  options: ValidateOptions = {},
): ValidationReport<Label> {
  const k = options.k ?? 5;
  const random = createRandom(options.seed ?? 1);

  const byLabel = new Map<Label, Example<Label>[]>();
  for (const example of examples) {
    const bucket = byLabel.get(example.label);
    if (bucket === undefined) byLabel.set(example.label, [example]);
    else bucket.push(example);
  }
  const smallestSupport = Math.min(...Array.from(byLabel.values(), (bucket) => bucket.length));
  const folds = Math.max(2, Math.min(options.folds ?? 5, Number.isFinite(smallestSupport) ? smallestSupport : 2));

  if (examples.length < 2 || byLabel.size < 2) {
    return {
      accuracy: 0,
      folds: 0,
      total: examples.length,
      labels: Array.from(byLabel, ([label, bucket]) => ({
        label,
        support: bucket.length,
        recall: 0,
        precision: 0,
      })),
      confusion: {},
    };
  }

  // Stratified assignment: shuffle each label's bucket, then deal round-robin.
  const assignment = new Map<string, number>();
  for (const bucket of byLabel.values()) {
    const shuffled = shuffle(bucket, random);
    for (let i = 0; i < shuffled.length; i += 1) {
      assignment.set((shuffled[i] as Example<Label>).id, i % folds);
    }
  }

  const confusion: Record<string, Record<string, number>> = {};
  let correct = 0;
  let evaluated = 0;
  for (let fold = 0; fold < folds; fold += 1) {
    const train = examples.filter((example) => assignment.get(example.id) !== fold);
    const test = examples.filter((example) => assignment.get(example.id) === fold);
    if (train.length === 0 || test.length === 0) continue;
    for (const example of test) {
      const prediction = predictWith(train, example.embedding, k);
      const predicted = prediction.label ?? '';
      const row = confusion[example.label] ?? {};
      row[predicted] = (row[predicted] ?? 0) + 1;
      confusion[example.label] = row;
      if (predicted === example.label) correct += 1;
      evaluated += 1;
    }
  }

  const labels: LabelReport<Label>[] = Array.from(byLabel, ([label, bucket]) => {
    const row = confusion[label] ?? {};
    const hits = row[label] ?? 0;
    const tested = Object.values(row).reduce((sum, count) => sum + count, 0);
    let predictedAnywhere = 0;
    for (const other of Object.values(confusion)) predictedAnywhere += other[label] ?? 0;
    return {
      label,
      support: bucket.length,
      recall: tested === 0 ? 0 : hits / tested,
      precision: predictedAnywhere === 0 ? 0 : hits / predictedAnywhere,
    };
  }).sort((a, b) => b.support - a.support);

  return {
    accuracy: evaluated === 0 ? 0 : correct / evaluated,
    folds,
    total: evaluated,
    labels,
    confusion,
  };
}

function shuffle<T>(items: readonly T[], random: () => number): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const a = out[i] as T;
    out[i] = out[j] as T;
    out[j] = a;
  }
  return out;
}
