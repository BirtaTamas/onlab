# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `16`
- rows: `111`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 111 | 1.000 | 0.613399 | 0.664562 | -0.051164 | 4 | 107 | 0.693694 | 1.000000 |
| active/recent utility | 111 | 1.000 | 0.613399 | 0.664562 | -0.051164 | 4 | 107 | 0.693694 | 1.000000 |
| strong utility action | 95 | 0.856 | 0.635435 | 0.682223 | -0.046788 | 4 | 91 | 0.789474 | 1.000000 |
| utility damage | 22 | 0.198 | 0.598814 | 0.670821 | -0.072007 | 0 | 22 | 0.454545 | 1.000000 |
| active smoke/inferno | 95 | 0.856 | 0.635435 | 0.682223 | -0.046788 | 4 | 91 | 0.789474 | 1.000000 |
| recent utility last 5s | 10 | 0.090 | 0.524982 | 0.551521 | -0.026539 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 111 | 1.000 | 0.613399 | 0.664562 | -0.051164 | 4 | 107 | 0.693694 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `55.0s`, rows `95`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.6296`, XGBoost `0.7685`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.6267`, XGBoost `0.7630`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6334`, XGBoost `0.7566`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6626`, XGBoost `0.7856`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6485`, XGBoost `0.7679`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6561`, XGBoost `0.7630`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6636`, XGBoost `0.7686`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.6703`, XGBoost `0.7710`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.6660`, XGBoost `0.7630`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6904`, XGBoost `0.7854`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `0`
