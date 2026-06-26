# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `4`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.640993 | 0.740186 | -0.099194 | 13 | 217 | 0.878261 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.640993 | 0.740186 | -0.099194 | 13 | 217 | 0.878261 | 1.000000 |
| strong utility action | 163 | 0.709 | 0.588474 | 0.714174 | -0.125700 | 4 | 159 | 0.938650 | 1.000000 |
| utility damage | 10 | 0.043 | 0.579945 | 0.727663 | -0.147718 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 163 | 0.709 | 0.588474 | 0.714174 | -0.125700 | 4 | 159 | 0.938650 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.640993 | 0.740186 | -0.099194 | 13 | 217 | 0.878261 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `90.0s`, rows `163`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `58.0`, LSTM `0.5198`, XGBoost `0.7257`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5420`, XGBoost `0.7460`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.5084`, XGBoost `0.7108`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5419`, XGBoost `0.7437`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5063`, XGBoost `0.7075`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5457`, XGBoost `0.7439`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5460`, XGBoost `0.7437`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5487`, XGBoost `0.7460`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5469`, XGBoost `0.7437`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5111`, XGBoost `0.7075`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
