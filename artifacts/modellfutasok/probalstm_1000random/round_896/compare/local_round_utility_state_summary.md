# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `1`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.688586 | 0.788213 | -0.099627 | 0 | 157 | 0.974522 | 1.000000 |
| active/recent utility | 157 | 1.000 | 0.688586 | 0.788213 | -0.099627 | 0 | 157 | 0.974522 | 1.000000 |
| strong utility action | 87 | 0.554 | 0.678024 | 0.770895 | -0.092871 | 0 | 87 | 0.977011 | 1.000000 |
| utility damage | 11 | 0.070 | 0.643529 | 0.747099 | -0.103570 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 87 | 0.554 | 0.678024 | 0.770895 | -0.092871 | 0 | 87 | 0.977011 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 157 | 1.000 | 0.688586 | 0.788213 | -0.099627 | 0 | 157 | 0.974522 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `53.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.0`, LSTM `0.5340`, XGBoost `0.7555`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5351`, XGBoost `0.7534`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5378`, XGBoost `0.7555`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5398`, XGBoost `0.7556`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5492`, XGBoost `0.7562`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5431`, XGBoost `0.7492`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5525`, XGBoost `0.7473`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5526`, XGBoost `0.7473`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `23.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5616`, XGBoost `0.7562`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5550`, XGBoost `0.7473`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `23.0`, recent_utility `0`
