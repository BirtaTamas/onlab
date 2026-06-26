# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `6`
- rows: `235`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.212705 | 0.296079 | -0.083374 | 216 | 19 | 0.991489 | 0.914894 |
| active/recent utility | 235 | 1.000 | 0.212705 | 0.296079 | -0.083374 | 216 | 19 | 0.991489 | 0.914894 |
| strong utility action | 139 | 0.591 | 0.226969 | 0.313072 | -0.086103 | 127 | 12 | 0.985612 | 0.856115 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 139 | 0.591 | 0.226969 | 0.313072 | -0.086103 | 127 | 12 | 0.985612 | 0.856115 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 235 | 1.000 | 0.212705 | 0.296079 | -0.083374 | 216 | 19 | 0.991489 | 0.914894 |

## Active Smoke/Inferno Intervals

- `7.0s` - `61.5s`, rows `110`
- `98.5s` - `112.5s`, rows `29`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `112.5`, LSTM `0.1601`, XGBoost `0.4969`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.3530`, XGBoost `0.6793`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.3713`, XGBoost `0.6800`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.4109`, XGBoost `0.6809`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.0860`, XGBoost `0.3423`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.4292`, XGBoost `0.6840`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.4328`, XGBoost `0.6827`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.4335`, XGBoost `0.6815`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.4009`, XGBoost `0.6428`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.4081`, XGBoost `0.6498`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `16.0`, recent_utility `0`
