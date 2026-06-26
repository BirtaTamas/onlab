# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `3`
- rows: `224`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.521748 | 0.433899 | 0.087849 | 166 | 58 | 0.584821 | 0.272321 |
| active/recent utility | 224 | 1.000 | 0.521748 | 0.433899 | 0.087849 | 166 | 58 | 0.584821 | 0.272321 |
| strong utility action | 207 | 0.924 | 0.513969 | 0.424914 | 0.089055 | 150 | 57 | 0.550725 | 0.212560 |
| utility damage | 20 | 0.089 | 0.499891 | 0.350227 | 0.149664 | 17 | 3 | 0.700000 | 0.000000 |
| active smoke/inferno | 207 | 0.924 | 0.513969 | 0.424914 | 0.089055 | 150 | 57 | 0.550725 | 0.212560 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.521748 | 0.433899 | 0.087849 | 166 | 58 | 0.584821 | 0.272321 |

## Active Smoke/Inferno Intervals

- `8.0s` - `111.0s`, rows `207`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.0`, LSTM `0.5569`, XGBoost `0.2555`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5550`, XGBoost `0.2555`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5561`, XGBoost `0.2576`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5540`, XGBoost `0.2555`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5565`, XGBoost `0.2585`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5564`, XGBoost `0.2585`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5412`, XGBoost `0.2440`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5410`, XGBoost `0.2440`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5538`, XGBoost `0.2568`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5457`, XGBoost `0.2540`, closer `lstm`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
