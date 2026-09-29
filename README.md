CurvTTC Three-Method Local Wrapper
===================================

This folder contains three CurvTTC calculation methods and one unified selector.

Files
-----
CurvTTC.py
    Unified public interface. Selects one of the three methods.

ContinuousSAT.py
    Continuous SAT-based approximation.

DiscreteSAT.py
    Discrete SAT-based reference method.

CenterTrajectory.py
    Center trajectory intersection method.

CalculateTestTTC.py
    Simple example for running a complete trajectory CSV.

How to use
----------
1. Put your trajectory CSV in this folder, for example:
       Test.csv

2. Open CalculateTestTTC.py.

3. Set:
       INPUT_CSV = "Test.csv"

4. Select one method:
       METHOD = "continuous"
       METHOD = "discrete"
       METHOD = "center"

5. Keep:
       OUTPUT_CSV = None

   Then the output name is generated automatically:
       continuous -> Test_ContinuousSAT.csv
       discrete   -> Test_DiscreteSAT.csv
       center     -> Test_CenterTrajectory.csv

You can also set an explicit output file:
       OUTPUT_CSV = "MyResult.csv"

Direct use from another Python file
-----------------------------------
from CurvTTC import calculate_curvttc

calculate_curvttc(
    input_csv="Test.csv",
    method="continuous",
    output_csv="Result.csv",
)

Available methods
-----------------
continuous
    Continuous SAT-based approximation.

discrete
    Discrete SAT-based reference method.

center
    Center trajectory intersection method.

Input columns
-------------
Required:
    count
    timestep
    AV_x
    AV_y
    Opp_x
    Opp_y
    AV_vx
    AV_vy
    Opp_vx
    Opp_vy

Optional vehicle dimensions:
    AV_length
    AV_width
    Opp_length
    Opp_width

If a dimension column is absent, the corresponding default value is used:
    AV:  4.50 m x 1.80 m
    Opp: 4.50 m x 1.80 m

Optional headings for SAT methods:
    AV_heading
    Opp_heading

Headings are interpreted in radians and are used by the SAT methods only when
the corresponding speed is numerically near zero. The center trajectory method
does not use heading.

Output
------
All three methods write the result column as:
    CurvTTC

A finite value is the predicted CurvTTC in seconds.
"inf" means no finite CurvTTC was identified within the modeled prediction range.
