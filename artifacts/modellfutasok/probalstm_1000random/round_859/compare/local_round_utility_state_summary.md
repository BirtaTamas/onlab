# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m3-nuke.csv`
- round_num: `11`
- rows: `199`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.570740 | 0.635854 | -0.065115 | 169 | 30 | 0.216080 | 0.115578 |
| active/recent utility | 199 | 1.000 | 0.570740 | 0.635854 | -0.065115 | 169 | 30 | 0.216080 | 0.115578 |
| strong utility action | 166 | 0.834 | 0.606357 | 0.682470 | -0.076113 | 143 | 23 | 0.144578 | 0.060241 |
| utility damage | 32 | 0.161 | 0.587922 | 0.676549 | -0.088627 | 32 | 0 | 0.281250 | 0.000000 |
| active smoke/inferno | 156 | 0.784 | 0.633958 | 0.715214 | -0.081255 | 138 | 18 | 0.089744 | 0.000000 |
| recent utility last 5s | 10 | 0.050 | 0.175779 | 0.171673 | 0.004106 | 5 | 5 | 1.000000 | 1.000000 |
| flash effect present | 199 | 1.000 | 0.570740 | 0.635854 | -0.065115 | 169 | 30 | 0.216080 | 0.115578 |

## Active Smoke/Inferno Intervals

- `8.0s` - `85.5s`, rows `156`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.5502`, XGBoost `0.7582`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6987`, XGBoost `0.9024`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7044`, XGBoost `0.9024`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5644`, XGBoost `0.7582`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5672`, XGBoost `0.7590`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5686`, XGBoost `0.7590`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5691`, XGBoost `0.7590`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.7249`, XGBoost `0.9127`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.7149`, XGBoost `0.9025`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.7194`, XGBoost `0.9028`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
