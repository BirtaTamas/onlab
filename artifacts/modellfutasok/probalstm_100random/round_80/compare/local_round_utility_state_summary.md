# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv`
- round_num: `15`
- rows: `262`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 262 | 1.000 | 0.355682 | 0.328386 | 0.027296 | 88 | 174 | 0.385496 | 0.400763 |
| active/recent utility | 262 | 1.000 | 0.355682 | 0.328386 | 0.027296 | 88 | 174 | 0.385496 | 0.400763 |
| strong utility action | 162 | 0.618 | 0.543110 | 0.500968 | 0.042143 | 2 | 160 | 0.055556 | 0.080247 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 154 | 0.588 | 0.541448 | 0.499691 | 0.041758 | 2 | 152 | 0.058442 | 0.084416 |
| recent utility last 5s | 10 | 0.038 | 0.577479 | 0.524665 | 0.052814 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 262 | 1.000 | 0.355682 | 0.328386 | 0.027296 | 88 | 174 | 0.385496 | 0.400763 |

## Active Smoke/Inferno Intervals

- `8.0s` - `84.5s`, rows `154`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.4169`, XGBoost `0.2478`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.3831`, XGBoost `0.2475`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.4014`, XGBoost `0.2715`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6423`, XGBoost `0.5278`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.3568`, XGBoost `0.2475`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6321`, XGBoost `0.5263`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.5327`, XGBoost `0.4273`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.5327`, XGBoost `0.4273`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6322`, XGBoost `0.5274`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.5326`, XGBoost `0.4283`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
