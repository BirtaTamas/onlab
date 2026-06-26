# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `6`
- rows: `151`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.753331 | 0.816534 | -0.063204 | 9 | 142 | 1.000000 | 1.000000 |
| active/recent utility | 151 | 1.000 | 0.753331 | 0.816534 | -0.063204 | 9 | 142 | 1.000000 | 1.000000 |
| strong utility action | 122 | 0.808 | 0.743324 | 0.817740 | -0.074416 | 5 | 117 | 1.000000 | 1.000000 |
| utility damage | 31 | 0.205 | 0.781860 | 0.840197 | -0.058338 | 2 | 29 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.808 | 0.743324 | 0.817740 | -0.074416 | 5 | 117 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 151 | 1.000 | 0.753331 | 0.816534 | -0.063204 | 9 | 142 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `39.5s`, rows `65`
- `42.5s` - `70.5s`, rows `57`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.6283`, XGBoost `0.8262`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6295`, XGBoost `0.8261`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.6346`, XGBoost `0.8258`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.6455`, XGBoost `0.8248`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6475`, XGBoost `0.8253`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6487`, XGBoost `0.8261`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.6490`, XGBoost `0.8249`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.6499`, XGBoost `0.8248`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.6516`, XGBoost `0.8248`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.6536`, XGBoost `0.8258`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
