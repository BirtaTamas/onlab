# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `6`
- rows: `191`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.528935 | 0.589804 | -0.060868 | 37 | 154 | 0.486911 | 0.863874 |
| active/recent utility | 191 | 1.000 | 0.528935 | 0.589804 | -0.060868 | 37 | 154 | 0.486911 | 0.863874 |
| strong utility action | 148 | 0.775 | 0.506789 | 0.572384 | -0.065595 | 33 | 115 | 0.466216 | 0.878378 |
| utility damage | 10 | 0.052 | 0.393446 | 0.516281 | -0.122835 | 0 | 10 | 0.000000 | 1.000000 |
| active smoke/inferno | 148 | 0.775 | 0.506789 | 0.572384 | -0.065595 | 33 | 115 | 0.466216 | 0.878378 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 191 | 1.000 | 0.528935 | 0.589804 | -0.060868 | 37 | 154 | 0.486911 | 0.863874 |

## Active Smoke/Inferno Intervals

- `9.5s` - `76.0s`, rows `134`
- `81.5s` - `88.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.3218`, XGBoost `0.5093`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `7.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3337`, XGBoost `0.5187`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.3355`, XGBoost `0.5196`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.3377`, XGBoost `0.5187`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.3294`, XGBoost `0.5076`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.2823`, XGBoost `0.4605`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3323`, XGBoost `0.5093`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `7.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3386`, XGBoost `0.5152`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `7.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.3342`, XGBoost `0.5088`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.3452`, XGBoost `0.5196`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
