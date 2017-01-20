# File Guide (further documentation within)

__Data ETL__ <br>
|── NBAParser.py = *Load, parse, store raw NBA json game files* <br>
__Train (Compute Graph)__ <br>
|── ComputeSynergies.py = *Compute Unweighted Synergy Graph* <br>
|── ComputeWeightedSynergies.py = *Compute Weighted Synergy Graph* <br>
__Predict (Predict based on Graph)__ <br>
|── PredictSynergy.py = *Predict using Unweighted Synergy Graph* <br>
|── PredictUnWeightedSynergy.py = *Predict using Weighted Synergy Graph* <br>
__Test Predictions__ <br>
|── predict_matchups.py = *Predict all matchups w/ test, train split* <br>
|── SynergyStepThrough.py = *Visualize how the Synergy Graph works* <br>
|── WeeklyPredictions.py = *Predict full games each week* <br>
__Utility Functions__ <br>
|── helper_functions.py = *Helper functions used throughout* <br>
|── combine_matchups.py = *Helper functions in preprocessing Data* <br>
|── home_away_table.py = *Ensure home, away is correct* <br>

