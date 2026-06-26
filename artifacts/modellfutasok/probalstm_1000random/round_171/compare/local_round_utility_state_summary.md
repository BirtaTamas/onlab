# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `10`
- rows: `150`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.007626 | 0.039014 | -0.031388 | 149 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 150 | 1.000 | 0.007626 | 0.039014 | -0.031388 | 149 | 1 | 1.000000 | 1.000000 |
| strong utility action | 108 | 0.720 | 0.006639 | 0.039026 | -0.032387 | 107 | 1 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.067 | 0.019114 | 0.107815 | -0.088701 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 108 | 0.720 | 0.006639 | 0.039026 | -0.032387 | 107 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 150 | 1.000 | 0.007626 | 0.039014 | -0.031388 | 149 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `60.5s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.0139`, XGBoost `0.1394`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.0116`, XGBoost `0.1333`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.0119`, XGBoost `0.1334`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0124`, XGBoost `0.1334`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0126`, XGBoost `0.1334`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.0130`, XGBoost `0.1333`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0123`, XGBoost `0.1324`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0143`, XGBoost `0.1341`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0124`, XGBoost `0.1242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0144`, XGBoost `0.1248`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `10.0`, recent_utility `0`
