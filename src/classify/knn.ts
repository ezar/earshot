/**
 * Cosine k-nearest-neighbours over YAMNet embeddings, with distance-weighted
 * votes and per-label prototypes.
 *
 * Meowlogue asks a user for ten examples per cat, not ten thousand. kNN over a
 * frozen embedding is the right tool at that scale: it needs no training, it
 * degrades gracefully with one example, and every prediction can point at the
 * stored examples that produced it.
 */

import { cosineDistance, l2Normalize } from '../util/math.js';

/** One labelled example. */
export interface Example<Label extends string = string> {
  /** Caller-supplied identifier, echoed back in neighbour lists. */
  readonly id: string;
  /** The label this example teaches. */
  readonly label: Label;
  /** Embedding vector; stored L2-normalized. */
  readonly embedding: readonly number[];
}

/** One neighbour behind a prediction. */
export interface Neighbour<Label extends string = string> {
  /** Id of the stored example. */
  readonly id: string;
  /** Its label. */
  readonly label: Label;
  /** Cosine distance to the query, in `[0, 2]`. */
  readonly distance: number;
  /** Weight this neighbour contributed to its label's vote. */
  readonly weight: number;
}

/** A ranked label with its share of the vote. */
export interface LabelScore<Label extends string = string> {
  readonly label: Label;
  /** Share of the total vote weight, in `[0, 1]`. */
  readonly score: number;
}

/** The outcome of classifying one embedding. */
export interface Prediction<Label extends string = string> {
  /** Best label; null when the classifier holds no examples. */
  readonly label: Label | null;
  /** Confidence of {@link label}, in `[0, 1]`. */
  readonly confidence: number;
  /** Every label that received a vote, strongest first. */
  readonly scores: readonly LabelScore<Label>[];
  /** The neighbours considered, nearest first. */
  readonly neighbours: readonly Neighbour<Label>[];
}

/** A label's averaged embedding. */
export interface Prototype<Label extends string = string> {
  readonly label: Label;
  /** Mean of the label's L2-normalized embeddings, itself normalized. */
  readonly embedding: readonly number[];
  /** How many examples went into it. */
  readonly count: number;
}

/** Serializable state of a {@link KnnClassifier}. */
export interface KnnSnapshot<Label extends string = string> {
  /** Shape version. */
  readonly schemaVersion: number;
  /** Stored examples, embeddings already normalized. */
  readonly examples: readonly Example<Label>[];
  /** Neighbours considered per query. */
  readonly k: number;
}

/** Schema version of {@link KnnSnapshot}. */
export const KNN_SCHEMA_VERSION = 1;

/** Options for {@link createKnnClassifier}. */
export interface KnnOptions {
  /** Neighbours to consider. Defaults to 5, capped at the example count. */
  readonly k?: number;
  /**
   * Neighbours further than this cosine distance are ignored. Defaults to 2,
   * i.e. no cut-off.
   */
  readonly maxDistance?: number;
}

/** A cosine kNN classifier over embeddings. */
export interface KnnClassifier<Label extends string = string> {
  /**
   * Add one labelled example.
   *
   * @param label - The label to teach.
   * @param embedding - Embedding vector; normalized on the way in.
   * @param id - Stable id; generated when omitted.
   * @returns The id under which the example was stored.
   */
  add(label: Label, embedding: ArrayLike<number>, id?: string): string;
  /** Remove one example by id; returns true when something was removed. */
  remove(id: string): boolean;
  /** Remove every example carrying `label`; returns how many were removed. */
  removeLabel(label: Label): number;
  /** Classify one embedding. */
  predict(embedding: ArrayLike<number>): Prediction<Label>;
  /** Per-label averaged embeddings. */
  prototypes(): Prototype<Label>[];
  /** Every stored example. */
  examples(): Example<Label>[];
  /** Labels currently known, in insertion order. */
  labels(): Label[];
  /** Serializable state. */
  toJSON(): KnnSnapshot<Label>;
  /** Replace the state from a snapshot. */
  load(snapshot: KnnSnapshot<Label>): void;
  /** Number of stored examples. */
  readonly size: number;
}

/**
 * Create an empty kNN classifier.
 *
 * @param options - Neighbour count and distance cut-off.
 */
export function createKnnClassifier<Label extends string = string>(
  options: KnnOptions = {},
): KnnClassifier<Label> {
  let k = options.k ?? 5;
  const maxDistance = options.maxDistance ?? 2;
  let examples: Example<Label>[] = [];
  let nextId = 0;

  return {
    add(label, embedding, id): string {
      const assigned = id ?? `e${nextId++}`;
      examples.push({ id: assigned, label, embedding: Array.from(l2Normalize(embedding)) });
      return assigned;
    },
    remove(id): boolean {
      const before = examples.length;
      examples = examples.filter((example) => example.id !== id);
      return examples.length < before;
    },
    removeLabel(label): number {
      const before = examples.length;
      examples = examples.filter((example) => example.label !== label);
      return before - examples.length;
    },
    predict(embedding): Prediction<Label> {
      return predictWith(examples, embedding, k, maxDistance);
    },
    prototypes(): Prototype<Label>[] {
      return buildPrototypes(examples);
    },
    examples(): Example<Label>[] {
      return examples.slice();
    },
    labels(): Label[] {
      const seen: Label[] = [];
      for (const example of examples) {
        if (!seen.includes(example.label)) seen.push(example.label);
      }
      return seen;
    },
    toJSON(): KnnSnapshot<Label> {
      return { schemaVersion: KNN_SCHEMA_VERSION, examples: examples.slice(), k };
    },
    load(snapshot): void {
      examples = snapshot.examples.map((example) => ({ ...example }));
      k = snapshot.k;
      nextId = examples.length;
    },
    get size(): number {
      return examples.length;
    },
  };
}

/**
 * Classify one embedding against a fixed example set.
 *
 * Votes are weighted by `1 / (distance + epsilon)`, so a near-identical example
 * dominates a merely similar one instead of each neighbour counting once.
 *
 * @param examples - Stored examples with normalized embeddings.
 * @param embedding - Query vector.
 * @param k - Neighbours to consider.
 * @param maxDistance - Cosine distance beyond which a neighbour is ignored.
 */
export function predictWith<Label extends string = string>(
  examples: readonly Example<Label>[],
  embedding: ArrayLike<number>,
  k: number,
  maxDistance = 2,
): Prediction<Label> {
  if (examples.length === 0) {
    return { label: null, confidence: 0, scores: [], neighbours: [] };
  }
  const query = l2Normalize(embedding);
  const ranked = examples
    .map((example) => ({ example, distance: cosineDistance(query, example.embedding) }))
    .filter((entry) => entry.distance <= maxDistance)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, Math.max(1, Math.min(k, examples.length)));

  if (ranked.length === 0) {
    return { label: null, confidence: 0, scores: [], neighbours: [] };
  }

  const votes = new Map<Label, number>();
  const neighbours: Neighbour<Label>[] = [];
  let total = 0;
  for (const entry of ranked) {
    const weight = 1 / (entry.distance + 1e-6);
    votes.set(entry.example.label, (votes.get(entry.example.label) ?? 0) + weight);
    total += weight;
    neighbours.push({
      id: entry.example.id,
      label: entry.example.label,
      distance: entry.distance,
      weight,
    });
  }

  const scores: LabelScore<Label>[] = Array.from(votes, ([label, weight]) => ({
    label,
    score: total > 0 ? weight / total : 0,
  })).sort((a, b) => b.score - a.score);

  const best = scores[0];
  return {
    label: best?.label ?? null,
    confidence: best?.score ?? 0,
    scores,
    neighbours,
  };
}

/** Average each label's examples into a prototype vector. */
export function buildPrototypes<Label extends string = string>(
  examples: readonly Example<Label>[],
): Prototype<Label>[] {
  const sums = new Map<Label, { sum: Float64Array; count: number }>();
  for (const example of examples) {
    const existing = sums.get(example.label);
    const target = existing ?? { sum: new Float64Array(example.embedding.length), count: 0 };
    for (let i = 0; i < example.embedding.length; i += 1) {
      target.sum[i] = (target.sum[i] as number) + (example.embedding[i] as number);
    }
    target.count += 1;
    if (existing === undefined) sums.set(example.label, target);
  }
  return Array.from(sums, ([label, { sum, count }]) => ({
    label,
    embedding: Array.from(l2Normalize(sum)),
    count,
  }));
}

/** Classify by nearest prototype instead of nearest neighbours. */
export function predictByPrototype<Label extends string = string>(
  prototypes: readonly Prototype<Label>[],
  embedding: ArrayLike<number>,
): Prediction<Label> {
  if (prototypes.length === 0) return { label: null, confidence: 0, scores: [], neighbours: [] };
  const query = l2Normalize(embedding);
  const ranked = prototypes
    .map((prototype) => ({ prototype, distance: cosineDistance(query, prototype.embedding) }))
    .sort((a, b) => a.distance - b.distance);

  let total = 0;
  const weights = ranked.map((entry) => {
    const weight = 1 / (entry.distance + 1e-6);
    total += weight;
    return weight;
  });
  const scores: LabelScore<Label>[] = ranked.map((entry, i) => ({
    label: entry.prototype.label,
    score: total > 0 ? (weights[i] as number) / total : 0,
  }));
  const best = scores[0];
  return {
    label: best?.label ?? null,
    confidence: best?.score ?? 0,
    scores,
    neighbours: ranked.map((entry) => ({
      id: `prototype:${entry.prototype.label}`,
      label: entry.prototype.label,
      distance: entry.distance,
      weight: 1 / (entry.distance + 1e-6),
    })),
  };
}
