from CurvTTC import calculate_curvttc

# Input trajectory CSV
INPUT_CSV = "Test.csv"

# Choose one:
#   "continuous" -> Continuous SAT
#   "discrete"   -> Discrete SAT
#   "center"     -> Center trajectory method
METHOD = "continuous"

# Set to None to generate the output name automatically.
# Example:
#   Test_ContinuousSAT.csv
#   Test_DiscreteSAT.csv
#   Test_CenterTrajectory.csv
OUTPUT_CSV = None


if __name__ == "__main__":
    calculate_curvttc(
        input_csv=INPUT_CSV,
        method=METHOD,
        output_csv=OUTPUT_CSV,
    )
