# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `5`
- rows: `191`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.517665 | 0.489161 | 0.028504 | 139 | 52 | 0.649215 | 0.602094 |
| active/recent utility | 191 | 1.000 | 0.517665 | 0.489161 | 0.028504 | 139 | 52 | 0.649215 | 0.602094 |
| strong utility action | 167 | 0.874 | 0.512210 | 0.478000 | 0.034210 | 130 | 37 | 0.670659 | 0.604790 |
| utility damage | 10 | 0.052 | 0.505184 | 0.396512 | 0.108672 | 10 | 0 | 0.700000 | 0.000000 |
| active smoke/inferno | 167 | 0.874 | 0.512210 | 0.478000 | 0.034210 | 130 | 37 | 0.670659 | 0.604790 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 191 | 1.000 | 0.517665 | 0.489161 | 0.028504 | 139 | 52 | 0.649215 | 0.602094 |

## Active Smoke/Inferno Intervals

- `7.0s` - `65.5s`, rows `118`
- `67.0s` - `91.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.4370`, XGBoost `0.2945`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.4279`, XGBoost `0.2974`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.5100`, XGBoost `0.3853`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.5251`, XGBoost `0.4012`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.4161`, XGBoost `0.2937`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.5261`, XGBoost `0.4048`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.5231`, XGBoost `0.4034`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.8171`, XGBoost `0.6991`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.4094`, XGBoost `0.2937`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.5112`, XGBoost `0.4020`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `37.0`, recent_utility `0`
