# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

# ============================================================
# Global constants / tolerances
# ============================================================
EPS_GEOM = 1e-10   # geometry numeric guard
EPS_NORM = 1e-10   # for normalization

FRAME_DT = 0.1     # 10Hz -> 0.1s per frame

# Default vehicle sizes (AV & Opp)
L_A_default, W_A_default = 4.50, 1.80
L_B_default, W_B_default = 4.50, 1.80

# Turn angle cap: 90 deg
THETA_MAX_RAD = math.pi / 2.0

# Input/output files
INPUT_CSV = "Leftturn_opposite_pairs_Filter_v_recon.csv"
OUTPUT_CSV = "DiscreteSAT.csv"

# Columns required by this batch script
MOTION_COLUMNS = [
    "AV_x", "AV_y", "Opp_x", "Opp_y",
    "AV_vx", "AV_vy", "Opp_vx", "Opp_vy",
]
REQUIRED_COLUMNS = ["count", "timestep"] + MOTION_COLUMNS
NUMERIC_COLUMNS = ["timestep"] + MOTION_COLUMNS
OPTIONAL_HEADING_COLUMNS = ["AV_heading", "Opp_heading"]
OPTIONAL_DIMENSION_COLUMNS = [
    "AV_length", "AV_width", "Opp_length", "Opp_width"
]


# ============================================================
# Input validation
# ============================================================
def validate_input_dataframe(df):
    """
    Fail fast when required columns are missing or input values are invalid.

    Invalid values include:
      - blank / missing values
      - NaN
      - +inf / -inf
      - non-numeric text in numeric input columns
      - invalid heading values when optional heading columns are present
      - invalid vehicle dimensions when optional dimension columns are present

    AV_heading and Opp_heading are optional. If either column is present, it is
    interpreted in radians and validated as a finite numeric column.

    AV_length, AV_width, Opp_length, and Opp_width are optional. If present,
    they are validated as finite numeric values greater than zero. If absent,
    the corresponding default vehicle dimensions are used.

    The function returns a validated copy in which numeric columns have been
    converted to numeric dtype.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    # count may be numeric or a string identifier, but it cannot be blank.
    count_text = df["count"].astype("string")
    invalid_count = df["count"].isna() | count_text.str.strip().eq("").fillna(True)
    if invalid_count.any():
        bad_rows = df.index[invalid_count].tolist()
        preview = ", ".join(map(str, bad_rows[:20]))
        suffix = " ..." if len(bad_rows) > 20 else ""
        raise ValueError(
            f"Column 'count' contains blank or missing values at row indices: "
            f"{preview}{suffix}"
        )

    # Optional heading and vehicle-dimension columns are validated when present.
    present_heading_columns = [
        col for col in OPTIONAL_HEADING_COLUMNS if col in df.columns
    ]
    present_dimension_columns = [
        col for col in OPTIONAL_DIMENSION_COLUMNS if col in df.columns
    ]
    numeric_columns = (
        NUMERIC_COLUMNS + present_heading_columns + present_dimension_columns
    )

    # Convert blanks and non-numeric text to NaN, then reject every non-finite value.
    numeric_data = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = pd.DataFrame(
        np.isfinite(numeric_data.to_numpy(dtype=float)),
        index=numeric_data.index,
        columns=numeric_data.columns,
    )
    invalid_cells = ~finite_mask

    if invalid_cells.to_numpy().any():
        details = []
        invalid_positions = np.argwhere(invalid_cells.to_numpy())

        for row_pos, col_pos in invalid_positions[:20]:
            row_index = invalid_cells.index[row_pos]
            column = invalid_cells.columns[col_pos]
            original_value = df.at[row_index, column]
            count_value = df.at[row_index, "count"]
            timestep_value = df.at[row_index, "timestep"]
            details.append(
                f"row={row_index}, count={count_value!r}, "
                f"timestep={timestep_value!r}, "
                f"column={column}, value={original_value!r}"
            )

        total_invalid = len(invalid_positions)
        more = "\n  ..." if total_invalid > 20 else ""
        detail_text = "\n  ".join(details)

        raise ValueError(
            f"Detected {total_invalid} invalid input cell(s). "
            "Blank values, NaN, infinity, and non-numeric text are not allowed "
            "in numeric input columns.\n"
            f"  {detail_text}{more}"
        )

    for column in present_dimension_columns:
        invalid_dimension = numeric_data[column] <= 0.0
        if invalid_dimension.any():
            bad_rows = numeric_data.index[invalid_dimension].tolist()
            preview = ", ".join(map(str, bad_rows[:20]))
            suffix = " ..." if len(bad_rows) > 20 else ""
            raise ValueError(
                f"Column '{column}' must contain values greater than zero. "
                f"Invalid row indices: {preview}{suffix}"
            )

    validated = df.copy()
    validated[numeric_columns] = numeric_data
    return validated


def validate_vehicle_dimensions(L_A, W_A, L_B, W_B):
    dimensions = {
        "L_A": L_A, "W_A": W_A,
        "L_B": L_B, "W_B": W_B,
    }
    invalid = [
        f"{name}={value!r}"
        for name, value in dimensions.items()
        if (not math.isfinite(float(value))) or float(value) <= 0.0
    ]
    if invalid:
        raise ValueError(
            "Vehicle dimensions must be finite and greater than zero: "
            + ", ".join(invalid)
        )


# Validate default vehicle dimensions once before processing.
validate_vehicle_dimensions(
    L_A_default, W_A_default, L_B_default, W_B_default
)


# ============================================================
# Vec helpers (2D)
# ============================================================
def vnorm(v):
    return math.hypot(v[0], v[1])


def vnormalize(v, eps=EPS_NORM, heading_rad=None):
    """
    Normalize a velocity vector to obtain the vehicle longitudinal axis.

    If the speed is smaller than eps:
      - use heading_rad when it is available (radians, measured from +x);
      - otherwise use the default +x direction (1, 0).
    """
    s = vnorm(v)
    if s < eps:
        if heading_rad is not None:
            heading_rad = float(heading_rad)
            if not math.isfinite(heading_rad):
                raise ValueError("heading_rad must be finite when provided.")
            return (math.cos(heading_rad), math.sin(heading_rad))
        return (1.0, 0.0)
    return (v[0] / s, v[1] / s)


def rot_p90(u):
    # rotate +90° (CCW): (x,y)->(-y, x)  [LEFT normal]
    return (-u[1], u[0])


# ============================================================
# 1) Circle fitting from three consecutive AV positions
#    If fitting fails, use the line model for the current frame.
# ============================================================
def circle_from_three_points_strict(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    mid1 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    mid2 = ((x2 + x3) / 2.0, (y2 + y3) / 2.0)

    k1 = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > EPS_GEOM else float("inf")
    k2 = (y3 - y2) / (x3 - x2) if abs(x3 - x2) > EPS_GEOM else float("inf")

    if k1 == 0:
        k1p = float("inf")
    elif math.isinf(k1):
        k1p = 0.0
    else:
        k1p = -1.0 / k1

    if k2 == 0:
        k2p = float("inf")
    elif math.isinf(k2):
        k2p = 0.0
    else:
        k2p = -1.0 / k2

    if math.isinf(k1p):
        cx = mid1[0]
        cy = k2p * (cx - mid2[0]) + mid2[1]
    elif math.isinf(k2p):
        cx = mid2[0]
        cy = k1p * (cx - mid1[0]) + mid1[1]
    else:
        denom = k1p - k2p
        if abs(denom) <= EPS_GEOM:
            raise ValueError(
                "degenerate: perpendicular bisectors nearly parallel "
                "(near-collinear)."
            )
        cx = (
            k1p * mid1[0]
            - k2p * mid2[0]
            + mid2[1]
            - mid1[1]
        ) / denom
        cy = k1p * (cx - mid1[0]) + mid1[1]

    R = math.hypot(x1 - cx, y1 - cy)
    if (
        (not math.isfinite(cx))
        or (not math.isfinite(cy))
        or (not math.isfinite(R))
        or (R <= EPS_GEOM)
    ):
        raise ValueError(
            "degenerate: non-finite circle center/radius or R too small."
        )
    return cx, cy, R


# ============================================================
# 2) OBB-SAT intersection
#      Ay / By  : LONG axis (Length)
#      Ax / Bx  : SHORT axis (Width)
# ============================================================
def obb_sat_intersect(P_A, A_x, A_y, L_A, W_A,
                      P_B, B_x, B_y, L_B, W_B):
    T_x, T_y = P_B[0] - P_A[0], P_B[1] - P_A[1]
    axes = [A_x, A_y, B_x, B_y]

    halfLA, halfWA = 0.5 * L_A, 0.5 * W_A
    halfLB, halfWB = 0.5 * L_B, 0.5 * W_B

    for n in axes:
        d = abs(T_x * n[0] + T_y * n[1])

        r_A = (
            halfLA * abs(A_y[0] * n[0] + A_y[1] * n[1])
            + halfWA * abs(A_x[0] * n[0] + A_x[1] * n[1])
        )

        r_B = (
            halfLB * abs(B_y[0] * n[0] + B_y[1] * n[1])
            + halfWB * abs(B_x[0] * n[0] + B_x[1] * n[1])
        )

        if d > r_A + r_B:
            return False

    return True


# ============================================================
# 3A) Collision test at time t (arc AV + line Opp)
# ============================================================
def is_colliding_at_time_arc(
    t, t0,
    P_A0, speed_A, v_Ax_raw, v_Ay_raw, circle_params,
    P_B0, v_Bx_raw, v_By_raw,
    L_A, W_A, L_B, W_B,
    heading_A_rad=None,
    heading_B_rad=None,
    eps_vn=EPS_NORM,
):
    (x_A0, y_A0) = P_A0
    (c_x, c_y, R, theta_A0) = circle_params
    (x_B0, y_B0) = P_B0

    # Rotation direction from r x v
    rx, ry = x_A0 - c_x, y_A0 - c_y
    z = rx * v_Ay_raw - ry * v_Ax_raw
    sign = 1.0 if z >= 0 else -1.0

    omega = speed_A / R
    theta_dot = sign * omega
    tau = t - t0

    # AV arc prediction
    theta_A = theta_A0 + theta_dot * tau
    x_A = c_x + R * math.cos(theta_A)
    y_A = c_y + R * math.sin(theta_A)

    # AV tangential velocity
    vAx = -R * theta_dot * math.sin(theta_A)
    vAy = R * theta_dot * math.cos(theta_A)

    # Opp constant-velocity linear motion
    x_B = x_B0 + v_Bx_raw * tau
    y_B = y_B0 + v_By_raw * tau

    # OBB axes from velocities; heading is used only at near-zero speed.
    A_y = vnormalize(
        (vAx, vAy), eps=eps_vn, heading_rad=heading_A_rad
    )
    A_x = rot_p90(A_y)

    B_y = vnormalize(
        (v_Bx_raw, v_By_raw), eps=eps_vn, heading_rad=heading_B_rad
    )
    B_x = rot_p90(B_y)

    return obb_sat_intersect(
        (x_A, y_A), A_x, A_y, L_A, W_A,
        (x_B, y_B), B_x, B_y, L_B, W_B,
    )


# ============================================================
# 3B) Collision test at time t (line AV + line Opp)
# ============================================================
def is_colliding_at_time_linear(
    t, t0,
    P_A0, v_Ax, v_Ay,
    P_B0, v_Bx, v_By,
    L_A, W_A, L_B, W_B,
    heading_A_rad=None,
    heading_B_rad=None,
    eps_vn=EPS_NORM,
):
    (x_A0, y_A0) = P_A0
    (x_B0, y_B0) = P_B0
    tau = t - t0

    x_A = x_A0 + v_Ax * tau
    y_A = y_A0 + v_Ay * tau
    x_B = x_B0 + v_Bx * tau
    y_B = y_B0 + v_By * tau

    A_y = vnormalize(
        (v_Ax, v_Ay), eps=eps_vn, heading_rad=heading_A_rad
    )
    A_x = rot_p90(A_y)

    B_y = vnormalize(
        (v_Bx, v_By), eps=eps_vn, heading_rad=heading_B_rad
    )
    B_x = rot_p90(B_y)

    return obb_sat_intersect(
        (x_A, y_A), A_x, A_y, L_A, W_A,
        (x_B, y_B), B_x, B_y, L_B, W_B,
    )


# ============================================================
# 4A) Discrete SAT: arc AV + line Opp
# ============================================================
def ttc_SAT_TOI_CCD_arc(
    P_A0, speed_A, v_Ax_raw, v_Ay_raw, circle_params,
    P_B0, v_Bx_raw, v_By_raw,
    t0,
    L_A=4.5, W_A=1.8, L_B=4.5, W_B=1.8,
    T_horizon=10.0,
    dt_bracket=0.1,
    tol=1e-4,
    max_bisect_iter=80,
    eps_vn=EPS_NORM,
    heading_A_rad=None,
    heading_B_rad=None,
):
    # A zero/near-zero AV cannot define an arc direction from velocity.
    if speed_A < eps_vn:
        return ttc_SAT_TOI_CCD_linear(
            P_A0=P_A0,
            v_Ax=v_Ax_raw, v_Ay=v_Ay_raw,
            P_B0=P_B0,
            v_Bx=v_Bx_raw, v_By=v_By_raw,
            t0=t0,
            L_A=L_A, W_A=W_A,
            L_B=L_B, W_B=W_B,
            T_horizon=T_horizon,
            dt_bracket=dt_bracket,
            tol=tol,
            max_bisect_iter=max_bisect_iter,
            eps_vn=eps_vn,
            heading_A_rad=heading_A_rad,
            heading_B_rad=heading_B_rad,
        )

    (c_x, c_y, R, theta_A0) = circle_params
    if abs(R) <= EPS_GEOM:
        return float("inf")

    # Already colliding at t = t0 => CurvTTC = 0
    if is_colliding_at_time_arc(
        t0, t0,
        P_A0, speed_A, v_Ax_raw, v_Ay_raw, circle_params,
        P_B0, v_Bx_raw, v_By_raw,
        L_A, W_A, L_B, W_B,
        heading_A_rad=heading_A_rad,
        heading_B_rad=heading_B_rad,
        eps_vn=eps_vn,
    ):
        return 0.0

    omega = speed_A / R
    if omega <= EPS_GEOM:
        return float("inf")

    T_eff = min(T_horizon, THETA_MAX_RAD / omega)
    if T_eff <= 0.0:
        return float("inf")

    N = math.ceil(T_eff / dt_bracket)
    t_lo = t0
    t_hi = None

    # Discrete scan to find the first intersecting sample.
    for k in range(1, N + 1):
        tau = min(k * dt_bracket, T_eff)
        t = t0 + tau

        if is_colliding_at_time_arc(
            t, t0,
            P_A0, speed_A, v_Ax_raw, v_Ay_raw, circle_params,
            P_B0, v_Bx_raw, v_By_raw,
            L_A, W_A, L_B, W_B,
            heading_A_rad=heading_A_rad,
            heading_B_rad=heading_B_rad,
            eps_vn=eps_vn,
        ):
            t_hi = t
            t_lo = t0 + max(0.0, (k - 1) * dt_bracket)
            break

    if t_hi is None:
        return float("inf")

    if is_colliding_at_time_arc(
        t_lo, t0,
        P_A0, speed_A, v_Ax_raw, v_Ay_raw, circle_params,
        P_B0, v_Bx_raw, v_By_raw,
        L_A, W_A, L_B, W_B,
        heading_A_rad=heading_A_rad,
        heading_B_rad=heading_B_rad,
        eps_vn=eps_vn,
    ):
        t_lo = t0

    # Refine the TOI by bisection.
    for _ in range(max_bisect_iter):
        if (t_hi - t_lo) <= tol:
            break

        t_mid = 0.5 * (t_lo + t_hi)

        if is_colliding_at_time_arc(
            t_mid, t0,
            P_A0, speed_A, v_Ax_raw, v_Ay_raw, circle_params,
            P_B0, v_Bx_raw, v_By_raw,
            L_A, W_A, L_B, W_B,
            heading_A_rad=heading_A_rad,
            heading_B_rad=heading_B_rad,
            eps_vn=eps_vn,
        ):
            t_hi = t_mid
        else:
            t_lo = t_mid

    return t_hi - t0


# ============================================================
# 4B) Discrete SAT: line AV + line Opp
# ============================================================
def ttc_SAT_TOI_CCD_linear(
    P_A0, v_Ax, v_Ay,
    P_B0, v_Bx, v_By,
    t0,
    L_A=4.5, W_A=1.8, L_B=4.5, W_B=1.8,
    T_horizon=10.0,
    dt_bracket=0.1,
    tol=1e-4,
    max_bisect_iter=80,
    eps_vn=EPS_NORM,
    heading_A_rad=None,
    heading_B_rad=None,
):
    # Already colliding at t = t0 => CurvTTC = 0
    if is_colliding_at_time_linear(
        t0, t0,
        P_A0, v_Ax, v_Ay,
        P_B0, v_Bx, v_By,
        L_A, W_A, L_B, W_B,
        heading_A_rad=heading_A_rad,
        heading_B_rad=heading_B_rad,
        eps_vn=eps_vn,
    ):
        return 0.0

    if T_horizon <= 0.0:
        return float("inf")

    N = math.ceil(T_horizon / dt_bracket)
    t_lo = t0
    t_hi = None

    # Discrete scan to find the first intersecting sample.
    for k in range(1, N + 1):
        tau = min(k * dt_bracket, T_horizon)
        t = t0 + tau

        if is_colliding_at_time_linear(
            t, t0,
            P_A0, v_Ax, v_Ay,
            P_B0, v_Bx, v_By,
            L_A, W_A, L_B, W_B,
            heading_A_rad=heading_A_rad,
            heading_B_rad=heading_B_rad,
            eps_vn=eps_vn,
        ):
            t_hi = t
            t_lo = t0 + max(0.0, (k - 1) * dt_bracket)
            break

    if t_hi is None:
        return float("inf")

    if is_colliding_at_time_linear(
        t_lo, t0,
        P_A0, v_Ax, v_Ay,
        P_B0, v_Bx, v_By,
        L_A, W_A, L_B, W_B,
        heading_A_rad=heading_A_rad,
        heading_B_rad=heading_B_rad,
        eps_vn=eps_vn,
    ):
        t_lo = t0

    # Refine the TOI by bisection.
    for _ in range(max_bisect_iter):
        if (t_hi - t_lo) <= tol:
            break

        t_mid = 0.5 * (t_lo + t_hi)

        if is_colliding_at_time_linear(
            t_mid, t0,
            P_A0, v_Ax, v_Ay,
            P_B0, v_Bx, v_By,
            L_A, W_A, L_B, W_B,
            heading_A_rad=heading_A_rad,
            heading_B_rad=heading_B_rad,
            eps_vn=eps_vn,
        ):
            t_hi = t_mid
        else:
            t_lo = t_mid

    return t_hi - t0


# ============================================================
# 5) Main: read -> validate -> group -> compute CurvTTC
#    Rule:
#      - first 2 frames: AV line + Opp line
#      - from frame 3 on: AV arc + Opp line
#      - if 3-point circle fitting fails: AV line + Opp line fallback
# ============================================================


def _show_inf(x):
    return (
        "inf"
        if isinstance(x, (float, np.floating)) and math.isinf(float(x))
        else x
    )


# ============================================================
# 5) Batch calculation
# ============================================================
def calculate_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = validate_input_dataframe(df)
    df["CurvTTC"] = pd.NA

    for count, group in df.groupby("count", sort=False):
        g = group.sort_values("timestep")
        orig_idx = g.index.tolist()

        if len(g) == 0:
            continue

        ttc_values = []

        for i in range(len(g)):
            idx_i = g.index[i]

            # Current positions
            P_A0 = (float(g.loc[idx_i, "AV_x"]), float(g.loc[idx_i, "AV_y"]))
            P_B0 = (float(g.loc[idx_i, "Opp_x"]), float(g.loc[idx_i, "Opp_y"]))

            # Current velocities
            v_Ax_raw = float(g.loc[idx_i, "AV_vx"])
            v_Ay_raw = float(g.loc[idx_i, "AV_vy"])
            v_Bx_raw = float(g.loc[idx_i, "Opp_vx"])
            v_By_raw = float(g.loc[idx_i, "Opp_vy"])

            speed_A = math.hypot(v_Ax_raw, v_Ay_raw)
            t0 = float(g.loc[idx_i, "timestep"]) * FRAME_DT

            # Optional current-frame headings in radians. They are used only when
            # the corresponding speed magnitude is below EPS_NORM.
            heading_A_rad = (
                float(g.loc[idx_i, "AV_heading"])
                if "AV_heading" in g.columns
                else None
            )
            heading_B_rad = (
                float(g.loc[idx_i, "Opp_heading"])
                if "Opp_heading" in g.columns
                else None
            )

            # Optional vehicle dimensions; use defaults when columns are absent.
            L_A = (
                float(g.loc[idx_i, "AV_length"])
                if "AV_length" in g.columns else L_A_default
            )
            W_A = (
                float(g.loc[idx_i, "AV_width"])
                if "AV_width" in g.columns else W_A_default
            )
            L_B = (
                float(g.loc[idx_i, "Opp_length"])
                if "Opp_length" in g.columns else L_B_default
            )
            W_B = (
                float(g.loc[idx_i, "Opp_width"])
                if "Opp_width" in g.columns else W_B_default
            )

            if i < 2:
                # First two frames: AV treated as a straight line.
                TTC = ttc_SAT_TOI_CCD_linear(
                    P_A0=P_A0,
                    v_Ax=v_Ax_raw, v_Ay=v_Ay_raw,
                    P_B0=P_B0,
                    v_Bx=v_Bx_raw, v_By=v_By_raw,
                    t0=t0,
                    L_A=L_A, W_A=W_A,
                    L_B=L_B, W_B=W_B,
                    T_horizon=10.0,
                    dt_bracket=0.1,
                    tol=1e-4,
                    heading_A_rad=heading_A_rad,
                    heading_B_rad=heading_B_rad,
                )
            else:
                idx_i1 = g.index[i - 1]
                idx_i2 = g.index[i - 2]

                p1 = (float(g.loc[idx_i2, "AV_x"]), float(g.loc[idx_i2, "AV_y"]))
                p2 = (float(g.loc[idx_i1, "AV_x"]), float(g.loc[idx_i1, "AV_y"]))
                p3 = P_A0

                try:
                    c_x, c_y, R = circle_from_three_points_strict(p1, p2, p3)
                except ValueError:
                    TTC = ttc_SAT_TOI_CCD_linear(
                        P_A0=P_A0,
                        v_Ax=v_Ax_raw, v_Ay=v_Ay_raw,
                        P_B0=P_B0,
                        v_Bx=v_Bx_raw, v_By=v_By_raw,
                        t0=t0,
                        L_A=L_A, W_A=W_A,
                        L_B=L_B, W_B=W_B,
                        T_horizon=10.0,
                        dt_bracket=0.1,
                        tol=1e-4,
                        heading_A_rad=heading_A_rad,
                        heading_B_rad=heading_B_rad,
                    )
                else:
                    theta_A0 = math.atan2(p3[1] - c_y, p3[0] - c_x)

                    TTC = ttc_SAT_TOI_CCD_arc(
                        P_A0=p3,
                        speed_A=speed_A,
                        v_Ax_raw=v_Ax_raw, v_Ay_raw=v_Ay_raw,
                        circle_params=(c_x, c_y, R, theta_A0),
                        P_B0=P_B0,
                        v_Bx_raw=v_Bx_raw, v_By_raw=v_By_raw,
                        t0=t0,
                        L_A=L_A, W_A=W_A,
                        L_B=L_B, W_B=W_B,
                        T_horizon=10.0,
                        dt_bracket=0.1,
                        tol=1e-4,
                        heading_A_rad=heading_A_rad,
                        heading_B_rad=heading_B_rad,
                    )

            ttc_values.append(TTC)

        # ttc_values and orig_idx have the same length.
        df.loc[orig_idx, "CurvTTC"] = ttc_values

    df["CurvTTC"] = df["CurvTTC"].astype(object).apply(_show_inf)
    df.to_csv(output_csv, index=False)

    print(f"Generated: {output_csv}")
    print(
        "Number of frames with finite CurvTTC:",
        ((df["CurvTTC"].notna()) & (df["CurvTTC"] != "inf")).sum(),
    )
    return df


def main():
    return calculate_csv(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
