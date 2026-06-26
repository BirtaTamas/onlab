# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `19`
- rows: `115`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 115 | 1.000 | 0.745701 | 0.819948 | -0.074247 | 1 | 114 | 0.791304 | 1.000000 |
| active/recent utility | 115 | 1.000 | 0.745701 | 0.819948 | -0.074247 | 1 | 114 | 0.791304 | 1.000000 |
| strong utility action | 97 | 0.843 | 0.799205 | 0.864040 | -0.064835 | 1 | 96 | 0.927835 | 1.000000 |
| utility damage | 21 | 0.183 | 0.845617 | 0.886390 | -0.040773 | 0 | 21 | 0.952381 | 1.000000 |
| active smoke/inferno | 93 | 0.809 | 0.793403 | 0.858936 | -0.065533 | 1 | 92 | 0.924731 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 115 | 1.000 | 0.745701 | 0.819948 | -0.074247 | 1 | 114 | 0.791304 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `54.5s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.5`, LSTM `0.3332`, XGBoost `0.5402`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.3388`, XGBoost `0.5405`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3393`, XGBoost `0.5327`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3609`, XGBoost `0.5202`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.3758`, XGBoost `0.5342`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.3816`, XGBoost `0.5207`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.3999`, XGBoost `0.5245`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.7604`, XGBoost `0.8832`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6164`, XGBoost `0.7215`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7803`, XGBoost `0.8832`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
