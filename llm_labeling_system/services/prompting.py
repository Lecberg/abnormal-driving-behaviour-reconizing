from __future__ import annotations

import json
from typing import Any


LABELS = ["normal", "speeding", "harsh_accel_brake", "zigzag_unstable", "unclear"]
RISK_LEVELS = ["low", "medium", "high", "unclear"]


SYSTEM_PROMPT = """
You are labeling short vehicle GPS trajectory windows for driving behavior research.

Return valid JSON only. Do not return Markdown.

Your task is to judge the driving behavior from the evidence in one trajectory window.

Important rules:
1. Do not label only by one fixed threshold.
2. Do not recreate the old rule-based label system.
3. Use multiple signals together: speed level, speed change, heading change, brake signal, turn signals, road/context changes, vehicle state, time sequence, and data quality.
4. If the evidence is weak, missing, noisy, or contradictory, use "unclear".
5. Do not guess road speed limits unless road speed-limit data is provided.
6. Prefer conservative labels. A wrong abnormal label is worse than an "unclear" label.
7. Explain the decision in plain language.

Allowed labels:
- normal: no clear abnormal driving pattern.
- speeding: speed appears unusually high or unsafe based on the available context.
- harsh_accel_brake: sudden acceleration, sudden braking, large speed change, or brake signal pattern.
- zigzag_unstable: repeated direction changes, unstable heading movement, or zigzag/tortuous path.
- unclear: not enough reliable evidence to judge.

Return exactly this JSON shape:
{
  "label": "normal | speeding | harsh_accel_brake | zigzag_unstable | unclear",
  "confidence": 0.0,
  "risk_level": "low | medium | high | unclear",
  "evidence": ["short evidence item 1", "short evidence item 2"],
  "reason": "short plain explanation",
  "data_quality_flags": [],
  "use_for_training": true,
  "human_review_needed": true
}

Rules for use_for_training:
- false if label is "unclear"
- false if confidence is below 0.55
- false if important data is missing
- true only when the label is reasonably supported
"""


def build_user_prompt(window: dict[str, Any]) -> str:
    compact_window = {
        "window_id": window["window_id"],
        "vehicle_id": window["vehicle_id"],
        "start_time": window["start_time"],
        "end_time": window["end_time"],
        "point_count": window["point_count"],
        "summary": window["summary"],
        "rows": window["rows"],
    }
    return (
        "Label this trajectory window.\n\n"
        "Use the summary and raw rows together.\n"
        "Do not use one signal alone unless the evidence is very strong.\n"
        "If the case is ambiguous, choose \"unclear\".\n\n"
        "Trajectory window:\n"
        + json.dumps(compact_window, ensure_ascii=False)
    )


def validate_label_response(payload: dict[str, Any]) -> dict[str, Any]:
    label = str(payload.get("label", "unclear")).strip()
    if label not in LABELS:
        label = "unclear"

    risk_level = str(payload.get("risk_level", "unclear")).strip()
    if risk_level not in RISK_LEVELS:
        risk_level = "unclear"

    confidence = payload.get("confidence", 0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(1.0, confidence_value))

    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(item).strip() for item in evidence if str(item).strip()][:8]

    data_quality_flags = payload.get("data_quality_flags", [])
    if not isinstance(data_quality_flags, list):
        data_quality_flags = [str(data_quality_flags)]
    data_quality_flags = [str(item).strip() for item in data_quality_flags if str(item).strip()][:8]

    reason = str(payload.get("reason", "")).strip()
    use_for_training = bool(payload.get("use_for_training", label != "unclear" and confidence_value >= 0.55))
    if label == "unclear" or confidence_value < 0.55:
        use_for_training = False

    human_review_needed = bool(payload.get("human_review_needed", True))
    if label == "unclear" or confidence_value < 0.8 or data_quality_flags:
        human_review_needed = True

    return {
        "label": label,
        "confidence": round(confidence_value, 4),
        "risk_level": risk_level,
        "evidence": evidence,
        "reason": reason,
        "data_quality_flags": data_quality_flags,
        "use_for_training": use_for_training,
        "human_review_needed": human_review_needed,
    }


def mock_label(window: dict[str, Any]) -> dict[str, Any]:
    summary = window.get("summary", {})
    flags = list(summary.get("data_quality_flags", []))
    max_speed = summary.get("max_gps_speed") or 0
    total_heading_change = summary.get("total_heading_change") or 0
    brake_count = summary.get("brake_count") or 0

    if flags:
        label = "unclear"
        risk = "unclear"
        confidence = 0.35
        reason = "Mock label: data quality flags make the window hard to judge."
    elif total_heading_change > 180:
        label = "zigzag_unstable"
        risk = "medium"
        confidence = 0.64
        reason = "Mock label: heading changes are large across the window."
    elif brake_count >= 2:
        label = "harsh_accel_brake"
        risk = "medium"
        confidence = 0.61
        reason = "Mock label: multiple brake signals appear in a short window."
    elif max_speed >= 90:
        label = "speeding"
        risk = "medium"
        confidence = 0.6
        reason = "Mock label: maximum GPS speed is high."
    else:
        label = "normal"
        risk = "low"
        confidence = 0.58
        reason = "Mock label: no strong abnormal pattern was found."

    return validate_label_response(
        {
            "label": label,
            "confidence": confidence,
            "risk_level": risk,
            "evidence": ["mock_mode", f"max_speed={max_speed}", f"total_heading_change={total_heading_change}"],
            "reason": reason,
            "data_quality_flags": flags,
            "use_for_training": label != "unclear",
        }
    )
