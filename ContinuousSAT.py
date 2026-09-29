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
OUTPUT_CSV = "AV_leftturn_opposite_pairs_ContinuousSAT.csv"

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
def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1]

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

def outer_radius(L, W):
    return 0.5 * math.sqrt(L * L + W * W)


# ============================================================
# 1) Circle fitting from three consecutive AV positions
#    If fitting fails, use the line model for the current frame.
# ============================================================
def circle_from_three_points_strict(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    mid1 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    mid2 = ((x2 + x3) / 2.0, (y2 + y3) / 2.0)

    k1 = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > EPS_GEOM else float('inf')
    k2 = (y3 - y2) / (x3 - x2) if abs(x3 - x2) > EPS_GEOM else float('inf')

    if k1 == 0:
        k1p = float('inf')
    elif math.isinf(k1):
        k1p = 0.0
    else:
        k1p = -1.0 / k1

    if k2 == 0:
        k2p = float('inf')
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
        denom = (k1p - k2p)
        if abs(denom) <= EPS_GEOM:
            raise ValueError("degenerate: perpendicular bisectors nearly parallel (near-collinear).")
        cx = (k1p * mid1[0] - k2p * mid2[0] + mid2[1] - mid1[1]) / denom
        cy = k1p * (cx - mid1[0]) + mid1[1]

    R = math.hypot(x1 - cx, y1 - cy)
    if (not math.isfinite(cx)) or (not math.isfinite(cy)) or (not math.isfinite(R)) or (R <= EPS_GEOM):
        raise ValueError("degenerate: non-finite circle center/radius or R too small.")
    return cx, cy, R


# ============================================================
# 2) OBB–SAT intersection
#      Ay / By  : LONG axis (Length)
#      Ax / Bx  : SHORT axis (Width)
# ============================================================
def obb_sat_intersect(P_A, A_x, A_y, L_A, W_A,
                      P_B, B_x, B_y, L_B, W_B):
    # Translation from A to B
    T_x, T_y = P_B[0] - P_A[0], P_B[1] - P_A[1]
    axes = [A_x, A_y, B_x, B_y]

    halfLA, halfWA = 0.5 * L_A, 0.5 * W_A
    halfLB, halfWB = 0.5 * L_B, 0.5 * W_B

    for n in axes:
        # center distance projected on axis n
        d = abs(T_x * n[0] + T_y * n[1])

        # projection radii on axis n (L along *_y, W along *_x)
        r_A = halfLA * abs(A_y[0]*n[0] + A_y[1]*n[1]) + \
              halfWA * abs(A_x[0]*n[0] + A_x[1]*n[1])

        r_B = halfLB * abs(B_y[0]*n[0] + B_y[1]*n[1]) + \
              halfWB * abs(B_x[0]*n[0] + B_x[1]*n[1])

        if d > r_A + r_B:
            return False
    return True


# ============================================================
# 3) Collision-at-time (relative time tau from CURRENT FRAME = 0)
#    Arc AV + Line Opp, axes from velocities with heading fallback
#    Ax/Bx are LEFT normals (CCW +90°)
# ============================================================
def is_colliding_at_tau_arc_vdir(tau,
                                 P_A0, v_Ax_raw, v_Ay_raw, circle_params,
                                 P_B0, v_Bx_raw, v_By_raw,
                                 L_A, W_A, L_B, W_B,
                                 heading_A_rad=None, heading_B_rad=None,
                                 eps_vn=EPS_NORM):
    (x_A0, y_A0) = P_A0
    (c_x, c_y, R, theta_A0) = circle_params
    (x_B0, y_B0) = P_B0

    # Determine rotation direction from r x v at tau=0
    rx, ry = x_A0 - c_x, y_A0 - c_y
    z = rx * v_Ay_raw - ry * v_Ax_raw
    sign = 1.0 if z >= 0 else -1.0

    speed_A = math.hypot(v_Ax_raw, v_Ay_raw)
    omega = speed_A / R
    theta_dot = sign * omega

    # AV arc state at tau
    theta_A = theta_A0 + theta_dot * tau
    x_A = c_x + R * math.cos(theta_A)
    y_A = c_y + R * math.sin(theta_A)

    # AV tangent velocity from geometry
    vAx = -R * theta_dot * math.sin(theta_A)
    vAy =  R * theta_dot * math.cos(theta_A)

    # Opp line state at tau
    x_B = x_B0 + v_Bx_raw * tau
    y_B = y_B0 + v_By_raw * tau

    # OBB axes from velocities; heading is used only at near-zero speed
    A_y = vnormalize(
        (vAx, vAy), eps=eps_vn, heading_rad=heading_A_rad
    )
    A_x = rot_p90(A_y)  # LEFT normal

    B_y = vnormalize(
        (v_Bx_raw, v_By_raw), eps=eps_vn, heading_rad=heading_B_rad
    )
    B_x = rot_p90(B_y)  # LEFT normal

    return obb_sat_intersect(
        (x_A, y_A), A_x, A_y, L_A, W_A,
        (x_B, y_B), B_x, B_y, L_B, W_B
    )


# 4) Frozen Continuous SAT
#    tau: prediction time from the current frame (tau = 0)
#    Step1: outer-circle trigger -> tau0 (freeze time)
#    Step2: freeze at tau0
#    Step3: continuous SAT on [tau0, t_end]
def ttc_FROZEN_CCD_arc_vdir_rel(
    P_A0, v_Ax_raw, v_Ay_raw, circle_params,
    P_B0, v_Bx_raw, v_By_raw,
    L_A=4.5, W_A=1.8, L_B=4.5, W_B=1.8,
    T_horizon=10.0,
    scan_dt=0.1,
    bisect_iters=10,
    eps_vn=EPS_NORM,
    heading_A_rad=None,
    heading_B_rad=None
):
    (c_x, c_y, R, theta_A0) = circle_params
    if abs(R) <= EPS_GEOM:
        return float('inf')

    speed_A = math.hypot(v_Ax_raw, v_Ay_raw)

    # A zero/near-zero AV cannot define an arc direction from velocity. Treat it
    # with the line model; its OBB orientation comes from AV_heading when
    # present, otherwise from the default +x direction.
    if speed_A < eps_vn:
        return ttc_FROZEN_CCD_line_vdir_rel(
            P_A0=P_A0,
            v_Ax_raw=v_Ax_raw, v_Ay_raw=v_Ay_raw,
            P_B0=P_B0,
            v_Bx_raw=v_Bx_raw, v_By_raw=v_By_raw,
            L_A=L_A, W_A=W_A,
            L_B=L_B, W_B=W_B,
            T_horizon=T_horizon,
            eps_vn=eps_vn,
            heading_A_rad=heading_A_rad,
            heading_B_rad=heading_B_rad,
        )

    omega = speed_A / R
    if omega <= EPS_GEOM:
        return float('inf')

    # effective horizon limited by 90 deg
    t_end = min(T_horizon, THETA_MAX_RAD / omega)
    if t_end <= 0.0:
        return float('inf')

    # already colliding at tau=0 => TTC=0
    if is_colliding_at_tau_arc_vdir(
        0.0, P_A0, v_Ax_raw, v_Ay_raw, circle_params,
        P_B0, v_Bx_raw, v_By_raw,
        L_A, W_A, L_B, W_B,
        heading_A_rad=heading_A_rad,
        heading_B_rad=heading_B_rad,
        eps_vn=eps_vn,
    ):
        return 0.0

    # rotation direction from r x v at tau=0
    (x_A0, y_A0) = P_A0
    rx, ry = x_A0 - c_x, y_A0 - c_y
    z = rx * v_Ay_raw - ry * v_Ax_raw
    sign = 1.0 if z >= 0 else -1.0
    theta_dot = sign * omega

    def PA_and_vA(tau):
        theta_A = theta_A0 + theta_dot * tau
        xA = c_x + R * math.cos(theta_A)
        yA = c_y + R * math.sin(theta_A)
        vAx = -R * theta_dot * math.sin(theta_A)
        vAy =  R * theta_dot * math.cos(theta_A)
        return xA, yA, vAx, vAy

    def PB_at(tau):
        xB0, yB0 = P_B0
        return (xB0 + v_Bx_raw * tau, yB0 + v_By_raw * tau)

    # Step1: find earliest freeze time by outer-circle contact
    rA = outer_radius(L_A, W_A)
    rB = outer_radius(L_B, W_B)
    Rsum2 = (rA + rB) ** 2

    def g(tau):
        xA, yA, _, _ = PA_and_vA(tau)
        xB, yB = PB_at(tau)
        dx, dy = xB - xA, yB - yA
        return dx*dx + dy*dy - Rsum2

    if g(0.0) <= 0.0:
        tau0 = 0.0
    else:
        tau0 = None
        tauL = 0.0
        kmax = int(t_end / scan_dt) + 1
        for k in range(1, kmax + 1):
            tauR = min(k * scan_dt, t_end)
            if g(tauR) <= 0.0:
                # bisect in [tauL, tauR]
                a, b = tauL, tauR
                for _ in range(bisect_iters):
                    m = 0.5 * (a + b)
                    if g(m) <= 0.0:
                        b = m
                    else:
                        a = m
                tau0 = b
                break
            tauL = tauR
            if tauR >= t_end - 1e-15:
                break

        if tau0 is None:
            return float('inf')

    # Step2: freeze at tau0
    xA, yA, vAx, vAy = PA_and_vA(tau0)
    xB, yB = PB_at(tau0)

    # Relative displacement: T0 = PA - PB  (at freeze)
    T0x, T0y = (xA - xB), (yA - yB)

    # Frozen axes from velocities with heading fallback, Ax/Bx are LEFT normals
    A_y = vnormalize(
        (vAx, vAy), eps=eps_vn, heading_rad=heading_A_rad
    )
    A_x = rot_p90(A_y)

    B_y = vnormalize(
        (v_Bx_raw, v_By_raw), eps=eps_vn, heading_rad=heading_B_rad
    )
    B_x = rot_p90(B_y)

    axes = [A_x, A_y, B_x, B_y]

    # Frozen relative velocity
    vrelx, vrely = (vAx - v_Bx_raw), (vAy - v_By_raw)

    halfLA, halfWA = 0.5 * L_A, 0.5 * W_A
    halfLB, halfWB = 0.5 * L_B, 0.5 * W_B

    # Length along *_y; width along *_x
    def R_A(n):
        return halfLA * abs(vdot(A_y, n)) + halfWA * abs(vdot(A_x, n))

    def R_B(n):
        return halfLB * abs(vdot(B_y, n)) + halfWB * abs(vdot(B_x, n))

    # Step3: interval SAT on tau in [tau0, t_end]
    t_first = tau0
    t_last  = t_end

    for n in axes:
        Rn = R_A(n) + R_B(n)
        s0 = T0x * n[0] + T0y * n[1]
        vn = vrelx * n[0] + vrely * n[1]

        if abs(vn) < eps_vn:
            # no relative motion along this axis
            if abs(s0) > Rn:
                return float('inf')
            continue

        # Solve |s0 + vn*(t - tau0)| <= Rn  =>  t in [t1, t2]
        t1 = tau0 + (-Rn - s0) / vn
        t2 = tau0 + ( Rn - s0) / vn
        enter = min(t1, t2)
        exit_ = max(t1, t2)

        # clamp to [tau0, t_end]
        if enter < tau0:
            enter = tau0
        if exit_ > t_end:
            exit_ = t_end

        if enter > exit_:
            return float('inf')

        t_first = max(t_first, enter)
        t_last  = min(t_last, exit_)

        if t_first > t_last:
            return float('inf')

    # current frame time is 0 => TTC = t_first
    return max(0.0, t_first)


# ============================================================
# 5) Continuous SAT for line vs line
#    Used for the first two frames.
#    OBB axes are obtained from velocity, with heading fallback.
# ============================================================
def ttc_FROZEN_CCD_line_vdir_rel(
    P_A0, v_Ax_raw, v_Ay_raw,
    P_B0, v_Bx_raw, v_By_raw,
    L_A=4.5, W_A=1.8, L_B=4.5, W_B=1.8,
    T_horizon=10.0,
    eps_vn=EPS_NORM,
    heading_A_rad=None,
    heading_B_rad=None
):
    # OBB axes from velocities; use heading only when speed is below eps_vn.
    A_y = vnormalize(
        (v_Ax_raw, v_Ay_raw), eps=eps_vn, heading_rad=heading_A_rad
    )
    A_x = rot_p90(A_y)

    B_y = vnormalize(
        (v_Bx_raw, v_By_raw), eps=eps_vn, heading_rad=heading_B_rad
    )
    B_x = rot_p90(B_y)

    axes = [A_x, A_y, B_x, B_y]

    # Relative displacement: T0 = PA - PB
    T0x, T0y = (P_A0[0] - P_B0[0]), (P_A0[1] - P_B0[1])

    # Relative velocity
    vrelx, vrely = (v_Ax_raw - v_Bx_raw), (v_Ay_raw - v_By_raw)

    halfLA, halfWA = 0.5 * L_A, 0.5 * W_A
    halfLB, halfWB = 0.5 * L_B, 0.5 * W_B

    def R_A(n):
        return halfLA * abs(vdot(A_y, n)) + halfWA * abs(vdot(A_x, n))

    def R_B(n):
        return halfLB * abs(vdot(B_y, n)) + halfWB * abs(vdot(B_x, n))

    # Already colliding at t=0 => TTC=0
    if obb_sat_intersect(
        P_A0, A_x, A_y, L_A, W_A,
        P_B0, B_x, B_y, L_B, W_B
    ):
        return 0.0

    t_first = 0.0
    t_last = T_horizon

    for n in axes:
        Rn = R_A(n) + R_B(n)
        s0 = T0x * n[0] + T0y * n[1]
        vn = vrelx * n[0] + vrely * n[1]

        if abs(vn) < eps_vn:
            if abs(s0) > Rn:
                return float('inf')
            continue

        # Solve |s0 + vn*t| <= Rn  =>  t in [t1, t2]
        t1 = (-Rn - s0) / vn
        t2 = ( Rn - s0) / vn
        enter = min(t1, t2)
        exit_ = max(t1, t2)

        # clamp to [0, T_horizon]
        if enter < 0.0:
            enter = 0.0
        if exit_ > T_horizon:
            exit_ = T_horizon

        if enter > exit_:
            return float('inf')

        t_first = max(t_first, enter)
        t_last = min(t_last, exit_)

        if t_first > t_last:
            return float('inf')

    return max(0.0, t_first)


# ============================================================
# 6) Main: read -> validate -> group -> compute TTC
#    Rule:
#      - first 2 frames: AV line + Opp line
#      - from frame 3 on: AV arc + Opp line
#      - if 3-point circle fitting fails: AV line + Opp line fallback
# ============================================================


def _show_inf(x):
    return "inf" if isinstance(x, (float, np.floating)) and math.isinf(float(x)) else x


# ============================================================
# 6) Batch calculation
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
                TTC = ttc_FROZEN_CCD_line_vdir_rel(
                    P_A0=P_A0,
                    v_Ax_raw=v_Ax_raw, v_Ay_raw=v_Ay_raw,
                    P_B0=P_B0,
                    v_Bx_raw=v_Bx_raw, v_By_raw=v_By_raw,
                    L_A=L_A, W_A=W_A,
                    L_B=L_B, W_B=W_B,
                    T_horizon=10.0,
                    heading_A_rad=heading_A_rad,
                    heading_B_rad=heading_B_rad,
                )
            else:
                # Use three AV points to fit a circle. Only a circle-fitting
                # ValueError triggers the line-vs-line fallback.
                idx_i1 = g.index[i - 1]
                idx_i2 = g.index[i - 2]

                p1 = (float(g.loc[idx_i2, "AV_x"]), float(g.loc[idx_i2, "AV_y"]))
                p2 = (float(g.loc[idx_i1, "AV_x"]), float(g.loc[idx_i1, "AV_y"]))
                p3 = P_A0

                try:
                    c_x, c_y, R = circle_from_three_points_strict(p1, p2, p3)
                except ValueError:
                    TTC = ttc_FROZEN_CCD_line_vdir_rel(
                        P_A0=P_A0,
                        v_Ax_raw=v_Ax_raw, v_Ay_raw=v_Ay_raw,
                        P_B0=P_B0,
                        v_Bx_raw=v_Bx_raw, v_By_raw=v_By_raw,
                        L_A=L_A, W_A=W_A,
                        L_B=L_B, W_B=W_B,
                        T_horizon=10.0,
                        heading_A_rad=heading_A_rad,
                        heading_B_rad=heading_B_rad,
                    )
                else:
                    theta_A0 = math.atan2(p3[1] - c_y, p3[0] - c_x)

                    TTC = ttc_FROZEN_CCD_arc_vdir_rel(
                        P_A0=p3,
                        v_Ax_raw=v_Ax_raw, v_Ay_raw=v_Ay_raw,
                        circle_params=(c_x, c_y, R, theta_A0),
                        P_B0=P_B0,
                        v_Bx_raw=v_Bx_raw, v_By_raw=v_By_raw,
                        L_A=L_A, W_A=W_A,
                        L_B=L_B, W_B=W_B,
                        T_horizon=10.0,
                        scan_dt=0.1,
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
