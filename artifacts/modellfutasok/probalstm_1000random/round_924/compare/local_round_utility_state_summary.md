# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `4`
- rows: `137`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.664267 | 0.712935 | -0.048668 | 51 | 86 | 1.000000 | 1.000000 |
| active/recent utility | 137 | 1.000 | 0.664267 | 0.712935 | -0.048668 | 51 | 86 | 1.000000 | 1.000000 |
| strong utility action | 101 | 0.737 | 0.680283 | 0.699144 | -0.018861 | 48 | 53 | 1.000000 | 1.000000 |
| utility damage | 17 | 0.124 | 0.715689 | 0.716804 | -0.001115 | 8 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 91 | 0.664 | 0.675817 | 0.696808 | -0.020991 | 42 | 49 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.080 | 0.719879 | 0.721273 | -0.001394 | 6 | 5 | 1.000000 | 1.000000 |
| flash effect present | 137 | 1.000 | 0.664267 | 0.712935 | -0.048668 | 51 | 86 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `51.5s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.5310`, XGBoost `0.6581`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5552`, XGBoost `0.6678`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5129`, XGBoost `0.6238`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5567`, XGBoost `0.6666`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5163`, XGBoost `0.6238`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5498`, XGBoost `0.6560`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5263`, XGBoost `0.6309`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5198`, XGBoost `0.6238`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5710`, XGBoost `0.6750`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5187`, XGBoost `0.6216`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
