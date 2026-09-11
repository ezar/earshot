/** Thin wrapper over MediaPipe's YAMNet `AudioClassifier`. */

import { SAMPLE_RATE_HZ } from '../constants.js';
import type { ClassScore } from '../util/types.js';
import { defaultTasksAudioLoader, type ModelUrls, type TasksAudioClassifier } from './tasks-audio.js';

/** Options for {@link createClassifier}. */
export interface ClassifierOptions extends ModelUrls {
  /** Maximum classes to keep per window. Defaults to 8. */
  readonly maxResults?: number;
  /** Scores below this are dropped. Defaults to 0.02. */
  readonly scoreThreshold?: number;
}

/** A loaded audio classifier. */
export interface Classifier {
  /**
   * Classify one analysis window.
   *
   * @param samples - Mono window at {@link SAMPLE_RATE_HZ}.
   * @returns Class scores, strongest first.
   */
  classify(samples: Float32Array): ClassScore[];
  /** Release the underlying WASM instance. */
  close(): void;
}

/**
 * Load the YAMNet classifier from app-provided URLs.
 *
 * @throws When `classifierUrl` is missing or the task file cannot be loaded.
 */
export async function createClassifier(options: ClassifierOptions): Promise<Classifier> {
  if (options.classifierUrl === undefined) {
    throw new Error('earshot: createClassifier requires models.classifierUrl');
  }
  const maxResults = options.maxResults ?? 8;
  const scoreThreshold = options.scoreThreshold ?? 0.02;
  const tasks = await (options.loadTasksAudio ?? defaultTasksAudioLoader)();
  const fileset = await tasks.FilesetResolver.forAudioTasks(options.wasmBaseUrl);
  const classifier: TasksAudioClassifier = await tasks.AudioClassifier.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: options.classifierUrl },
    maxResults,
    scoreThreshold,
  });

  return {
    classify(samples: Float32Array): ClassScore[] {
      const results = classifier.classify(samples, SAMPLE_RATE_HZ);
      return mergeClassifications(results, maxResults);
    },
    close(): void {
      classifier.close();
    },
  };
}

/**
 * Collapse MediaPipe's per-patch results into one ranked list.
 *
 * MediaPipe may split a window into several internal patches; earshot keeps the
 * highest score each class reached anywhere in the window, because a short
 * event should not be diluted by the quiet patches around it.
 */
export function mergeClassifications(
  results: readonly { readonly classifications: readonly { readonly categories: readonly { readonly categoryName?: string; readonly displayName?: string; readonly score: number }[] }[] }[],
  maxResults: number,
): ClassScore[] {
  const best = new Map<string, number>();
  for (const result of results) {
    for (const classification of result.classifications) {
      for (const category of classification.categories) {
        const label = category.categoryName ?? category.displayName ?? '';
        if (label === '') continue;
        const previous = best.get(label);
        if (previous === undefined || category.score > previous) best.set(label, category.score);
      }
    }
  }
  return Array.from(best, ([label, score]) => ({ label, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults);
}

/** Look up a class score by label; 0 when the class is not in the list. */
export function scoreOf(classes: readonly ClassScore[], label: string): number {
  for (const entry of classes) {
    if (entry.label === label) return entry.score;
  }
  return 0;
}

/** Highest score among `labels`; 0 when none of them are present. */
export function maxScoreOf(classes: readonly ClassScore[], labels: readonly string[]): number {
  let best = 0;
  for (const entry of classes) {
    if (labels.includes(entry.label) && entry.score > best) best = entry.score;
  }
  return best;
}
