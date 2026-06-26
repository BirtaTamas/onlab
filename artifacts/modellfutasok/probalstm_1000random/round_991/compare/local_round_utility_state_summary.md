# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `11`
- rows: `192`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 192 | 1.000 | 0.652398 | 0.634537 | 0.017861 | 120 | 72 | 0.973958 | 1.000000 |
| active/recent utility | 192 | 1.000 | 0.652398 | 0.634537 | 0.017861 | 120 | 72 | 0.973958 | 1.000000 |
| strong utility action | 153 | 0.797 | 0.636739 | 0.614460 | 0.022279 | 103 | 50 | 0.967320 | 1.000000 |
| utility damage | 11 | 0.057 | 0.646863 | 0.610973 | 0.035890 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 153 | 0.797 | 0.636739 | 0.614460 | 0.022279 | 103 | 50 | 0.967320 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 192 | 1.000 | 0.652398 | 0.634537 | 0.017861 | 120 | 72 | 0.973958 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `47.0s`, rows `75`
- `49.5s` - `88.0s`, rows `78`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.5`, LSTM `0.7939`, XGBoost `0.6261`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.6724`, XGBoost `0.5145`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.6456`, XGBoost `0.5145`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.6422`, XGBoost `0.5145`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.7313`, XGBoost `0.6060`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.6398`, XGBoost `0.5145`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.6358`, XGBoost `0.5145`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.7837`, XGBoost `0.6666`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.8597`, XGBoost `0.9695`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.7020`, XGBoost `0.5987`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
