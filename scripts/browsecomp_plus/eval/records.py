"""Result identities, resumability and complete mean@N metrics, without GPU imports."""

import json
from collections import Counter


def write_json(path, value):
    """Make each visible JSON file a complete record, including after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_record(path, index, sample, task_id):
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["index"], record["sample"], record["task_id"]) != (index, sample, str(task_id)):
        raise ValueError(f"Result identity mismatch: {path}")
    return record


def pending_items(rows, run, samples):
    """Keep every completed outcome, including wrong and unsubmitted answers."""
    pending = []
    for sample in range(samples):
        for index, row in enumerate(rows):
            path = run / "rollouts" / f"{index:03d}_{sample}.json"
            if path.exists():
                saved = read_record(path, index, sample, row["extra_info"]["task_id"])
                if saved["status"] == "completed":
                    continue
            pending.append((index, sample, row))
    return pending


def summarize(records, questions, samples, summary_limit):
    """Require full, unique task/sample coverage before reporting mean/pass@N."""
    expected = {(str(q), s) for q in questions for s in range(samples)}
    identities = {(r["task_id"], r["sample"]) for r in records}
    if len(records) != len(identities) or not identities <= expected:
        raise ValueError("Duplicate or unexpected judgment identities.")
    if identities != expected:
        return dict(
            complete=False,
            judged_rollouts=len(records),
            expected_rollouts=len(expected),
            pending_rollouts_or_judgments=len(expected) - len(records),
        )
    groups = {str(q): [] for q in questions}
    for row in records:
        assert row["reward"] in (0, 1)
        assert row["final_answer"] is not None or row["reward"] == 0
        groups[row["task_id"]].append(row)
    correct = Counter(sum(r["reward"] for r in group) for group in groups.values())
    answers = Counter(sum(r["final_answer"] is not None for r in group) for group in groups.values())
    submitted = sum(r["final_answer"] is not None for r in records)
    success = sum(r["reward"] for r in records)
    total = len(records)
    return dict(
        complete=True,
        questions=len(questions),
        rollouts=total,
        samples_per_question=samples,
        **{
            f"mean_at_{samples}": success / total,
            f"pass_at_{samples}": sum(any(r["reward"] for r in g) for g in groups.values()) / len(questions),
        },
        submission_rate=submitted / total,
        accuracy_when_submitted=success / submitted if submitted else None,
        tool_calls_mean=sum(r["tool_calls"] for r in records) / total,
        correct_count_distribution={str(i): correct[i] for i in range(samples + 1)},
        answer_count_distribution={str(i): answers[i] for i in range(samples + 1)},
        finish_reasons=dict(Counter(r["finish_reason"] for r in records)),
        turn_token_limit_rate=sum(bool(r["metadata"].get("turn_token_limit_hit")) for r in records) / total,
        # Summary truncation can be hidden by a later done/max_compactions outcome.
        summary_token_limit_rate=sum(
            any(n >= summary_limit for n in r["metadata"].get("compact_response_tokens", [])) for r in records
        )
        / total,
    )
