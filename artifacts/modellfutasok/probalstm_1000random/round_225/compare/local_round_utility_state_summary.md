# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `14`
- rows: `235`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.100596 | 0.156122 | -0.055526 | 207 | 28 | 1.000000 | 1.000000 |
| active/recent utility | 235 | 1.000 | 0.100596 | 0.156122 | -0.055526 | 207 | 28 | 1.000000 | 1.000000 |
| strong utility action | 183 | 0.779 | 0.110553 | 0.178646 | -0.068092 | 175 | 8 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.043 | 0.315029 | 0.467330 | -0.152301 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 171 | 0.728 | 0.093238 | 0.166580 | -0.073342 | 171 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 12 | 0.051 | 0.357290 | 0.350582 | 0.006709 | 4 | 8 | 1.000000 | 1.000000 |
| flash effect present | 235 | 1.000 | 0.100596 | 0.156122 | -0.055526 | 207 | 28 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `96.0s`, rows `171`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.1976`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.2021`, XGBoost `0.4429`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1993`, XGBoost `0.4398`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2000`, XGBoost `0.4398`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.2031`, XGBoost `0.4429`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2073`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2095`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.2061`, XGBoost `0.4398`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2098`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.2094`, XGBoost `0.4398`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
