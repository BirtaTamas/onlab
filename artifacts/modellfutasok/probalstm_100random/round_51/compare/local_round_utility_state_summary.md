# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `4`
- rows: `152`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 152 | 1.000 | 0.401238 | 0.463110 | -0.061872 | 121 | 31 | 0.690789 | 0.269737 |
| active/recent utility | 152 | 1.000 | 0.401238 | 0.463110 | -0.061872 | 121 | 31 | 0.690789 | 0.269737 |
| strong utility action | 112 | 0.737 | 0.440638 | 0.490277 | -0.049639 | 87 | 25 | 0.669643 | 0.196429 |
| utility damage | 20 | 0.132 | 0.293483 | 0.351292 | -0.057809 | 15 | 5 | 0.850000 | 0.600000 |
| active smoke/inferno | 102 | 0.671 | 0.469420 | 0.515090 | -0.045670 | 77 | 25 | 0.637255 | 0.117647 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 152 | 1.000 | 0.401238 | 0.463110 | -0.061872 | 121 | 31 | 0.690789 | 0.269737 |

## Active Smoke/Inferno Intervals

- `5.0s` - `55.5s`, rows `102`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.3361`, XGBoost `0.5196`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3455`, XGBoost `0.5211`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.3510`, XGBoost `0.5211`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3638`, XGBoost `0.5155`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.3681`, XGBoost `0.5176`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.3845`, XGBoost `0.5211`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.3821`, XGBoost `0.5187`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3835`, XGBoost `0.5187`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.3847`, XGBoost `0.5173`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3860`, XGBoost `0.5176`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
