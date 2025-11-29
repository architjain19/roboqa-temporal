import numpy as np

def compute_completeness(sequence_data, expected_freqs):
    metrics = {}
    values = []

    for topic, ts in sequence_data.items():
        if topic not in expected_freqs or len(ts) == 0:
            continue

        ts = np.array(ts)
        duration = max(ts[-1] - ts[0], 1e-6)

        expected = expected_freqs[topic] * duration
        actual = len(ts)

        completeness = min(actual / expected, 1.0)
        score = completeness * 100.0

        metrics[f"{topic}_completeness"] = score
        values.append(score)

    metrics["completeness_score"] = np.mean(values) if values else 0.0
    return metrics


def compute_temporal_consistency(sequence_data, expected_freqs):
    metrics = {}
    topic_scores = []

    for topic, ts in sequence_data.items():
        if topic not in expected_freqs or len(ts) < 3:
            continue

        ts = np.array(ts)
        diffs = np.diff(ts)
        nominal = 1.0 / expected_freqs[topic]

        std_ratio = np.std(diffs) / nominal
        max_gap = np.max(diffs)

        std_score = np.exp(-std_ratio)
        gap_score = np.exp(-(max_gap - 3 * nominal)) if max_gap > 3 * nominal else 1.0

        topic_score = (std_score + gap_score) / 2 * 100.0
        topic_scores.append(topic_score)

        metrics[f"{topic}_consistency_score"] = topic_score

    metrics["consistency_score"] = np.mean(topic_scores) if topic_scores else 0.0
    return metrics


def compute_timeliness(sequence_data):
    all_ts = [t for ts in sequence_data.values() for t in ts]
    if not all_ts:
        return {"timeliness_score": 0.0}

    all_ts = np.array(sorted(all_ts))
    duration = all_ts[-1] - all_ts[0]

    score = min(duration / 5.0, 1.0) * 100.0
    return {
        "duration_sec": duration,
        "timeliness_score": score,
    }
