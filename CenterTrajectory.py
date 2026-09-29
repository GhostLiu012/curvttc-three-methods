# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd


# ============================================================
# Global constants / tolerances
# ============================================================
EPS_GEOM = 1e-10
EPS_NORM = 1e-10
EPS_TIME = 1e-12

# Default vehicle sizes (AV & Opp)
L_LT_default, W_LT_default = 4.50, 1.80
L_Opp_default, W_Opp_default = 4.50, 1.80

# Prediction settings
T_HORIZON = 10.0
THETA_MAX_RAD = math.pi / 2.0

# Input/output files
INPUT_CSV = "Leftturn_opposite_pairs_Filter_v_recon.csv"
OUTPUT_CSV = "CenterTrajectory.csv"

# Columns required by this batch script
MOTION_COLUMNS = [
    "AV_x", "AV_y", "Opp_x", "Opp_y",
    "AV_vx", "AV_vy", "Opp_vx", "Opp_vy",
]
REQUIRED_COLUMNS = ["count", "timestep"] + MOTION_COLUMNS
NUMERIC_COLUMNS = ["timestep"] + MOTION_COLUMNS
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
      - invalid vehicle dimensions when optional dimension columns are present

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

    # Optional vehicle-dimension columns are validated when present.
    present_dimension_columns = [
        col for col in OPTIONAL_DIMENSION_COLUMNS if col in df.columns
    ]
    numeric_columns = NUMERIC_COLUMNS + present_dimension_columns

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


def validate_vehicle_dimensions(L_LT, W_LT, L_Opp, W_Opp):
    dimensions = {
        "L_LT": L_LT,
        "W_LT": W_LT,
        "L_Opp": L_Opp,
        "W_Opp": W_Opp,
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
    L_LT_default,
    W_LT_default,
    L_Opp_default,
    W_Opp_default,
)


# ============================================================
# Basic helpers
# ============================================================
def speed(vx, vy):
    return math.hypot(vx, vy)


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def forward_angular_displacement(theta0, theta_target, turn_sign):
    """
    Return the positive angular displacement from theta0 to theta_target
    along the turning direction.
    """
    two_pi = 2.0 * math.pi
    if turn_sign >= 0:
        return (theta_target - theta0) % two_pi
    return (theta0 - theta_target) % two_pi


def ttc_from_overlap(
    first_enter,
    first_exit,
    second_enter,
    second_exit,
    T_horizon,
):
    """Return the start of a valid future overlap interval, or infinity."""
    overlap_start = max(first_enter, second_enter)
    overlap_end = min(first_exit, second_exit)

    if overlap_start > overlap_end + EPS_TIME:
        return float("inf")

    if overlap_end < -EPS_TIME:
        return float("inf")

    ttc = max(0.0, overlap_start)
    if ttc <= T_horizon + EPS_TIME:
        return ttc

    return float("inf")


# ============================================================
# 1) Circle fitting from three consecutive AV positions
#    If fitting fails, use the line model for the current frame.
# ============================================================
def circle_from_three_points_strict(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    mid1 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    mid2 = ((x2 + x3) / 2.0, (y2 + y3) / 2.0)

    k1 = (
        (y2 - y1) / (x2 - x1)
        if abs(x2 - x1) > EPS_GEOM
        else float("inf")
    )
    k2 = (
        (y3 - y2) / (x3 - x2)
        if abs(x3 - x2) > EPS_GEOM
        else float("inf")
    )

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
# 2) Straight-line center trajectory method
#    Used for the first two frames and as a circle-fitting fallback.
# ============================================================
def ttc_center_trajectory_straight_line(
    P_LT0,
    v_LT,
    P_Opp0,
    v_Opp,
    L_LT=4.5,
    W_LT=1.8,
    L_Opp=4.5,
    W_Opp=1.8,
    T_horizon=10.0,
):
    """
    Compute CurvTTC using straight-line center trajectories for both vehicles.

    Time is relative to the current frame, where t = 0.
    """
    x_LT0, y_LT0 = P_LT0
    vx_LT0, vy_LT0 = v_LT
    x_Opp0, y_Opp0 = P_Opp0
    vx_Opp0, vy_Opp0 = v_Opp

    vLT_norm = speed(vx_LT0, vy_LT0)
    vOpp_norm = speed(vx_Opp0, vy_Opp0)

    if vLT_norm < EPS_NORM or vOpp_norm < EPS_NORM:
        return float("inf")

    half_dist_LT = 0.5 * L_LT + 0.5 * W_LT
    half_dist_Opp = 0.5 * L_Opp + 0.5 * W_Opp

    # Solve the line-line intersection.
    den = cross2d(v_LT, v_Opp)
    if abs(den) <= EPS_GEOM:
        return float("inf")

    rel = (x_Opp0 - x_LT0, y_Opp0 - y_LT0)
    t_LT = cross2d(rel, v_Opp) / den
    t_Opp = cross2d(rel, v_LT) / den

    half_time_LT = half_dist_LT / vLT_norm
    half_time_Opp = half_dist_Opp / vOpp_norm

    return ttc_from_overlap(
        t_LT - half_time_LT,
        t_LT + half_time_LT,
        t_Opp - half_time_Opp,
        t_Opp + half_time_Opp,
        T_horizon,
    )


# ============================================================
# 3) Circular-arc center trajectory method
#    Used from the third frame onward.
# ============================================================
def ttc_center_trajectory_conflict_zone(
    P_LT0,
    v_LT,
    circle_params,
    P_Opp0,
    v_Opp,
    L_LT=4.5,
    W_LT=1.8,
    L_Opp=4.5,
    W_Opp=1.8,
    T_horizon=10.0,
    theta_max_rad=math.pi / 2.0,
):
    """
    Compute CurvTTC using center-trajectory intersections and conflict zones.

    Time is relative to the current frame, where t = 0.
    """
    x_LT0, y_LT0 = P_LT0
    vx_LT0, vy_LT0 = v_LT
    c_x, c_y, R, theta_LT0 = circle_params
    x_Opp0, y_Opp0 = P_Opp0
    vx_Opp0, vy_Opp0 = v_Opp

    vLT_norm = speed(vx_LT0, vy_LT0)
    vOpp_norm = speed(vx_Opp0, vy_Opp0)

    if vLT_norm < EPS_NORM or vOpp_norm < EPS_NORM:
        return float("inf")

    if R <= EPS_GEOM:
        return float("inf")

    omega = vLT_norm / R
    if omega <= EPS_GEOM:
        return float("inf")

    # Determine the turning direction from r x v at the current frame.
    rx, ry = x_LT0 - c_x, y_LT0 - c_y
    cross_z = rx * vy_LT0 - ry * vx_LT0
    turn_sign = 1.0 if cross_z >= 0.0 else -1.0

    # Prediction horizon capped by both time and angular displacement.
    t_end = min(T_horizon, theta_max_rad / omega)
    if t_end <= 0.0:
        return float("inf")

    half_dist_LT = 0.5 * L_LT + 0.5 * W_LT
    half_dist_Opp = 0.5 * L_Opp + 0.5 * W_Opp
    half_time_LT = half_dist_LT / vLT_norm
    half_time_Opp = half_dist_Opp / vOpp_norm

    valid_ttc = []

    # Solve the line-circle intersection.
    dx0 = x_Opp0 - c_x
    dy0 = y_Opp0 - c_y

    alpha = vx_Opp0 * vx_Opp0 + vy_Opp0 * vy_Opp0
    beta = 2.0 * (dx0 * vx_Opp0 + dy0 * vy_Opp0)
    gamma = dx0 * dx0 + dy0 * dy0 - R * R

    disc = beta * beta - 4.0 * alpha * gamma
    if disc < -EPS_GEOM:
        return float("inf")

    if disc < 0.0:
        disc = 0.0

    sqrt_disc = math.sqrt(disc)
    roots = [(-beta - sqrt_disc) / (2.0 * alpha)]
    if sqrt_disc > EPS_GEOM:
        roots.append((-beta + sqrt_disc) / (2.0 * alpha))

    for t_con in roots:
        # t_con is the opposing vehicle center's arrival time at the conflict point.
        if t_con < -EPS_TIME:
            continue

        if t_con > t_end + max(half_time_LT, half_time_Opp) + EPS_TIME:
            continue

        if t_con < 0.0:
            t_con = 0.0

        x_con = x_Opp0 + vx_Opp0 * t_con
        y_con = y_Opp0 + vy_Opp0 * t_con

        theta_con = math.atan2(y_con - c_y, x_con - c_x)
        delta_theta = forward_angular_displacement(
            theta_LT0,
            theta_con,
            turn_sign,
        )
        T_con = delta_theta / omega

        # Exclude points not reached by LT within the modeled arc.
        if T_con > t_end + half_time_LT + EPS_TIME:
            continue

        opp_enter = t_con - half_time_Opp
        opp_exit = t_con + half_time_Opp
        lt_enter = T_con - half_time_LT
        lt_exit = T_con + half_time_LT

        ttc = ttc_from_overlap(
            opp_enter,
            opp_exit,
            lt_enter,
            lt_exit,
            t_end,
        )

        if math.isfinite(ttc):
            valid_ttc.append(ttc)

    if not valid_ttc:
        return float("inf")

    return min(valid_ttc)


# ============================================================
# 4) Main: read -> validate -> group -> compute CurvTTC
#    Rule:
#      - first 2 frames: AV line + Opp line
#      - from frame 3 on: AV arc + Opp line
#      - if 3-point circle fitting fails: AV line + Opp line fallback
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
            P_LT0 = (
                float(g.loc[idx_i, "AV_x"]),
                float(g.loc[idx_i, "AV_y"]),
            )
            P_Opp0 = (
                float(g.loc[idx_i, "Opp_x"]),
                float(g.loc[idx_i, "Opp_y"]),
            )

            # Current velocities
            v_LT = (
                float(g.loc[idx_i, "AV_vx"]),
                float(g.loc[idx_i, "AV_vy"]),
            )
            v_Opp = (
                float(g.loc[idx_i, "Opp_vx"]),
                float(g.loc[idx_i, "Opp_vy"]),
            )

            # Optional vehicle dimensions; use defaults when columns are absent.
            L_LT = (
                float(g.loc[idx_i, "AV_length"])
                if "AV_length" in g.columns else L_LT_default
            )
            W_LT = (
                float(g.loc[idx_i, "AV_width"])
                if "AV_width" in g.columns else W_LT_default
            )
            L_Opp = (
                float(g.loc[idx_i, "Opp_length"])
                if "Opp_length" in g.columns else L_Opp_default
            )
            W_Opp = (
                float(g.loc[idx_i, "Opp_width"])
                if "Opp_width" in g.columns else W_Opp_default
            )

            if i < 2:
                # First two frames: AV treated as a straight line.
                TTC = ttc_center_trajectory_straight_line(
                    P_LT0=P_LT0,
                    v_LT=v_LT,
                    P_Opp0=P_Opp0,
                    v_Opp=v_Opp,
                    L_LT=L_LT,
                    W_LT=W_LT,
                    L_Opp=L_Opp,
                    W_Opp=W_Opp,
                    T_horizon=T_HORIZON,
                )
            else:
                idx_i1 = g.index[i - 1]
                idx_i2 = g.index[i - 2]

                p1 = (
                    float(g.loc[idx_i2, "AV_x"]),
                    float(g.loc[idx_i2, "AV_y"]),
                )
                p2 = (
                    float(g.loc[idx_i1, "AV_x"]),
                    float(g.loc[idx_i1, "AV_y"]),
                )
                p3 = P_LT0

                try:
                    c_x, c_y, R = circle_from_three_points_strict(
                        p1,
                        p2,
                        p3,
                    )
                except ValueError:
                    TTC = ttc_center_trajectory_straight_line(
                        P_LT0=P_LT0,
                        v_LT=v_LT,
                        P_Opp0=P_Opp0,
                        v_Opp=v_Opp,
                        L_LT=L_LT,
                        W_LT=W_LT,
                        L_Opp=L_Opp,
                        W_Opp=W_Opp,
                        T_horizon=T_HORIZON,
                    )
                else:
                    theta_LT0 = math.atan2(
                        p3[1] - c_y,
                        p3[0] - c_x,
                    )

                    TTC = ttc_center_trajectory_conflict_zone(
                        P_LT0=p3,
                        v_LT=v_LT,
                        circle_params=(c_x, c_y, R, theta_LT0),
                        P_Opp0=P_Opp0,
                        v_Opp=v_Opp,
                        L_LT=L_LT,
                        W_LT=W_LT,
                        L_Opp=L_Opp,
                        W_Opp=W_Opp,
                        T_horizon=T_HORIZON,
                        theta_max_rad=THETA_MAX_RAD,
                    )

            ttc_values.append(TTC)

        # ttc_values and orig_idx have the same length.
        df.loc[orig_idx, "CurvTTC"] = ttc_values

    # ============================================================
    # 5) Export
    # ============================================================
    def _show_inf(x):
        return (
            "inf"
            if isinstance(x, (float, np.floating)) and math.isinf(float(x))
            else x
        )

    # Write positive infinity as "inf" in the output CSV.
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
