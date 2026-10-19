"""Shared v045 RCSS get-up actuator gains and mechanical torque caps.

The source-domain get-up gains could not hold a lightly perturbed upright RCSS
robot for three seconds.  The existing RCSS locomotion preparation gains did so
in 16/16 deterministic engineering trials.  v045 therefore uses those proven
gains for both training and deployment while retaining the lower competition
joint torque caps.
"""

from __future__ import annotations

import numpy as np


V045_GETUP_KP = np.asarray(
    [
        5.0, 5.0,
        40.0, 50.0, 20.0, 10.0,
        40.0, 50.0, 20.0, 10.0,
        100.0,
        350.0, 350.0, 180.0, 350.0, 450.0, 450.0,
        350.0, 350.0, 180.0, 350.0, 450.0, 450.0,
    ],
    dtype=np.float32,
)

V045_GETUP_KD = np.asarray(
    [
        0.1, 0.1,
        0.5, 1.5, 0.2, 0.2,
        0.5, 1.5, 0.2, 0.2,
        5.0,
        7.5, 7.5, 3.0, 5.5, 0.5, 0.5,
        7.5, 7.5, 3.0, 5.5, 0.5, 0.5,
    ],
    dtype=np.float32,
)

RCSS_EFFECTIVE_EFFORT_LIMIT = np.asarray(
    [
        7.0, 7.0,
        18.0, 18.0, 18.0, 18.0,
        18.0, 18.0, 18.0, 18.0,
        30.0,
        45.0, 30.0, 30.0, 60.0, 20.0, 15.0,
        45.0, 30.0, 30.0, 60.0, 20.0, 15.0,
    ],
    dtype=np.float32,
)

# MJLab patterns, Kp, Kd and effective effort cap.  The order is intentionally
# canonical so the compiled 23-servo snapshot can be compared element by element
# with deployment during the fail-closed target-model audit.
V045_ACTUATOR_GROUPS = (
    (("AAHead_yaw", "Head_pitch"), 5.0, 0.1, 7.0),
    ((".*_Shoulder_Pitch",), 40.0, 0.5, 18.0),
    ((".*_Shoulder_Roll",), 50.0, 1.5, 18.0),
    ((".*_Elbow_Pitch",), 20.0, 0.2, 18.0),
    ((".*_Elbow_Yaw",), 10.0, 0.2, 18.0),
    (("Waist",), 100.0, 5.0, 30.0),
    ((".*_Hip_Pitch",), 350.0, 7.5, 45.0),
    ((".*_Hip_Roll",), 350.0, 7.5, 30.0),
    ((".*_Hip_Yaw",), 180.0, 3.0, 30.0),
    ((".*_Knee_Pitch",), 350.0, 5.5, 60.0),
    ((".*_Ankle_Pitch",), 450.0, 0.5, 20.0),
    ((".*_Ankle_Roll",), 450.0, 0.5, 15.0),
)

for name, values in (
    ("V045_GETUP_KP", V045_GETUP_KP),
    ("V045_GETUP_KD", V045_GETUP_KD),
    ("RCSS_EFFECTIVE_EFFORT_LIMIT", RCSS_EFFECTIVE_EFFORT_LIMIT),
):
    if values.shape != (23,) or not np.isfinite(values).all():
        raise RuntimeError(f"{name} must be one finite 23D vector")
    if np.any(values <= 0.0):
        raise RuntimeError(f"{name} entries must all be positive")
