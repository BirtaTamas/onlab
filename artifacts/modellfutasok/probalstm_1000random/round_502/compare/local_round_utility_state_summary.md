# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `19`
- rows: `193`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 193 | 1.000 | 0.750089 | 0.772946 | -0.022858 | 42 | 151 | 0.943005 | 1.000000 |
| active/recent utility | 193 | 1.000 | 0.750089 | 0.772946 | -0.022858 | 42 | 151 | 0.943005 | 1.000000 |
| strong utility action | 141 | 0.731 | 0.716319 | 0.734674 | -0.018355 | 37 | 104 | 0.921986 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 133 | 0.689 | 0.716222 | 0.740477 | -0.024255 | 29 | 104 | 0.917293 | 1.000000 |
| recent utility last 5s | 10 | 0.052 | 0.719760 | 0.636460 | 0.083300 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 193 | 1.000 | 0.750089 | 0.772946 | -0.022858 | 42 | 151 | 0.943005 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `72.5s`, rows `133`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.3955`, XGBoost `0.5555`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4099`, XGBoost `0.5551`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4410`, XGBoost `0.5541`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.7316`, XGBoost `0.6294`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `37.0`, LSTM `0.4556`, XGBoost `0.5559`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4553`, XGBoost `0.5541`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.7355`, XGBoost `0.6376`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.7315`, XGBoost `0.6363`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.7227`, XGBoost `0.6294`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.7226`, XGBoost `0.6297`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `1`
