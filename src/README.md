Data ETL
    ├── NBAParser.py            # Load, parse, store raw NBA json game files
Train (Compute Graph)
    ├── ComputeSynergies.py     # Compute Unweighted Synergy Graph
    ├── ComputeWeightedSynergies.py  # Compute Weighted Synergy Graph
Predict (Predict based on Graph)  
    ├── PredictSynergy.py     # Predict using Unweighted Synergy Graph
    ├── PredictUnWeightedSynergy.py  # Predict using Weighted Synergy Graph
Test Predictions    
    ├── predict_matchups.py   # Predict all matchups w/ test, train split
    ├── SynergyStepThrough.py # Visualize how the Synergy Graph works
    ├── WeeklyPredictions.py  # Predict full games each week
Utility Functions
    ├── helper_functions.py   # Helper functions used throughout
    ├── combine_matchups.py   # Helper functions in preprocessing Data
    ├── home_away_table.py    # Ensure home, away is correct

