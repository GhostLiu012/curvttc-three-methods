# -*- coding: utf-8 -*-
from pathlib import Path

import ContinuousSAT
import DiscreteSAT
import CenterTrajectory


_METHODS = {
    "continuous": (ContinuousSAT.calculate_csv, "ContinuousSAT"),
    "continuoussat": (ContinuousSAT.calculate_csv, "ContinuousSAT"),
    "continuous_sat": (ContinuousSAT.calculate_csv, "ContinuousSAT"),

    "discrete": (DiscreteSAT.calculate_csv, "DiscreteSAT"),
    "discretesat": (DiscreteSAT.calculate_csv, "DiscreteSAT"),
    "discrete_sat": (DiscreteSAT.calculate_csv, "DiscreteSAT"),

    "center": (CenterTrajectory.calculate_csv, "CenterTrajectory"),
    "centertrajectory": (CenterTrajectory.calculate_csv, "CenterTrajectory"),
    "center_trajectory": (CenterTrajectory.calculate_csv, "CenterTrajectory"),
}


def calculate_curvttc(input_csv, method="continuous", output_csv=None):
    """
    Calculate CurvTTC for a complete trajectory CSV.

    Parameters
    ----------
    input_csv : str or Path
        Input trajectory CSV.
    method : str
        "continuous", "discrete", or "center".
    output_csv : str or Path, optional
        Output CSV. If omitted, a method-specific name is generated in the
        same folder as the input file.

    Returns
    -------
    pandas.DataFrame
        The processed dataframe containing the CurvTTC column.
    """
    method_key = str(method).strip().lower().replace("-", "_").replace(" ", "_")

    if method_key not in _METHODS:
        raise ValueError(
            "Unknown method. Choose one of: continuous, discrete, center."
        )

    calculate_function, suffix = _METHODS[method_key]

    input_path = Path(input_csv)

    if output_csv is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_{suffix}{input_path.suffix or '.csv'}"
        )
    else:
        output_path = Path(output_csv)

    return calculate_function(
        input_csv=str(input_path),
        output_csv=str(output_path),
    )
