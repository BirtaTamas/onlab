# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `18`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.602766 | 0.588189 | 0.014577 | 151 | 79 | 0.965217 | 0.508696 |
| active/recent utility | 230 | 1.000 | 0.602766 | 0.588189 | 0.014577 | 151 | 79 | 0.965217 | 0.508696 |
| strong utility action | 172 | 0.748 | 0.597384 | 0.581250 | 0.016134 | 110 | 62 | 0.953488 | 0.523256 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 169 | 0.735 | 0.598980 | 0.582896 | 0.016084 | 107 | 62 | 0.952663 | 0.532544 |
| recent utility last 5s | 10 | 0.043 | 0.516390 | 0.481336 | 0.035053 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.602766 | 0.588189 | 0.014577 | 151 | 79 | 0.965217 | 0.508696 |

## Active Smoke/Inferno Intervals

- `6.0s` - `68.0s`, rows `125`
- `84.5s` - `106.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `102.0`, LSTM `0.7597`, XGBoost `0.8548`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5563`, XGBoost `0.4744`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7614`, XGBoost `0.6860`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5533`, XGBoost `0.4787`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5464`, XGBoost `0.4739`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5493`, XGBoost `0.4797`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5380`, XGBoost `0.4739`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5435`, XGBoost `0.4795`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5372`, XGBoost `0.4745`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5591`, XGBoost `0.4966`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
