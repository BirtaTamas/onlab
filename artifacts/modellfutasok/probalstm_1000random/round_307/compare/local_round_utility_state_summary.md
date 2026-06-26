# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `16`
- rows: `311`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 311 | 1.000 | 0.445036 | 0.495955 | -0.050919 | 266 | 45 | 0.308682 | 0.305466 |
| active/recent utility | 311 | 1.000 | 0.445036 | 0.495955 | -0.050919 | 266 | 45 | 0.308682 | 0.305466 |
| strong utility action | 232 | 0.746 | 0.513495 | 0.613604 | -0.100109 | 228 | 4 | 0.107759 | 0.103448 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 222 | 0.714 | 0.508575 | 0.612807 | -0.104232 | 222 | 0 | 0.112613 | 0.108108 |
| recent utility last 5s | 10 | 0.032 | 0.622704 | 0.631294 | -0.008590 | 6 | 4 | 0.000000 | 0.000000 |
| flash effect present | 311 | 1.000 | 0.445036 | 0.495955 | -0.050919 | 266 | 45 | 0.308682 | 0.305466 |

## Active Smoke/Inferno Intervals

- `9.0s` - `119.5s`, rows `222`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `109.0`, LSTM `0.0434`, XGBoost `0.3782`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.0608`, XGBoost `0.3797`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.0307`, XGBoost `0.3186`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.0440`, XGBoost `0.3198`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.0570`, XGBoost `0.3133`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `108.0`, LSTM `0.1403`, XGBoost `0.3880`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.0445`, XGBoost `0.2579`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.0408`, XGBoost `0.2415`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.0395`, XGBoost `0.2289`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.0`, LSTM `0.0348`, XGBoost `0.2160`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
