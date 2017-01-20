import pandas as pd
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, add_date
from sklearn.cross_validation import train_test_split
from ComputeSynergies import ComputeSynergies
from ComputeWeightedSynergies import ComputeWeightedSynergies
from PredictSynergy import PredictSynergy
from PredictSynergyWeighted import PredictSynergyWeighted


def predict_all_matchups(syn_obj, df, season):
    """Predict each matchup using Synergy Graphs.

    Parameters
    ----------
    syn_obj : Computed Synergy Graph Object
        Can be Weighted or Unweighted.
    df : Pandas DataFrame
        df you want to predict.
    season : string, ex. '2008'
        season you are predicting on
    """
    predict_df = pd.DataFrame()
    for i in xrange(len(df)):
        matchup = df.iloc[i:i+1].reset_index(drop=True)
        if syn_obj.__module__ == "ComputeWeightedSynergies":
            ps = PredictSynergyWeighted(syn_obj, matchup)
        elif syn_obj.__module__ == "ComputeSynergies":
            ps = PredictSynergy(syn_obj, matchup)
        try:
            ps.predict_one()
        except KeyError:
            continue
        predict_df = predict_df.append(ps.by_entry)

    # com_df['correct'] = com_df['i_margin'] * com_df['prediction'] > 0
    # # com_df = com_df[com_df['prediction'].notnull()]

    return predict_df.reset_index(drop=True)


if __name__ == '__main__':
    season = '2008'
    X = read_season('matchups_reordered', season)
    X = add_date(X)
    # X = X[:int(len(X) * 0.3) + 1]
    all_preds = pd.DataFrame()
    k_folds = 1
    for k in xrange(k_folds):
        train_df, test_df = train_test_split(X, test_size=0.1)
        train_df = combine_same_matchups(train_df)
        # train_df = greater_than_minute(train_df)

        # Reset index on test set to make it easier to merge later
        test_df = test_df.reset_index(drop=True)

        # Compute and Predict for Unweighted Graph
        cs = ComputeSynergies(train_df)
        cs.initialize_random_graphs(10)
        # cs.simulated_annealing(200)
        preds = predict_all(cs, test_df, season)

        # Compute and Predict for Weighted Graph
        csw = ComputeWeightedSynergies(train_df)
        csw.genetic_algorithm(60, count=5)
        preds_w = predict_all(csw, test_df, '2008')

        # Merge both predictions
        com_preds = preds.merge(preds_w, left_index=True,
                                right_index=True, how='outer',
                                suffixes=("_u", "_w"))
        com_preds['k'] = k

        # Merge back test set data
        com_preds = pd.concat([test_df, com_preds], axis=1)
        all_preds = pd.concat([all_preds, com_preds])

    # all_preds['correct'] = all_preds['i_margin'] * all_preds['prediction_by_matchup']
    correct = (all_preds['prediction_by_matchup_w'] * all_preds['i_margin'] > 0).sum()
    incorrect = (all_preds['prediction_by_matchup_w'] * all_preds['i_margin'] < 0).sum()
    print "Percentage Correct = " + str(correct / float(correct + incorrect))

    all_preds.to_csv('../data/' + season + '_k_fold.csv')
