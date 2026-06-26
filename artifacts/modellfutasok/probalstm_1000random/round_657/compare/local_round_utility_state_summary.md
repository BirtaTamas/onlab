# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `4`
- rows: `260`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 260 | 1.000 | 0.590066 | 0.582116 | 0.007950 | 112 | 148 | 0.303846 | 0.223077 |
| active/recent utility | 260 | 1.000 | 0.590066 | 0.582116 | 0.007950 | 112 | 148 | 0.303846 | 0.223077 |
| strong utility action | 143 | 0.550 | 0.558702 | 0.567209 | -0.008506 | 82 | 61 | 0.405594 | 0.230769 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 143 | 0.550 | 0.558702 | 0.567209 | -0.008506 | 82 | 61 | 0.405594 | 0.230769 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 260 | 1.000 | 0.590066 | 0.582116 | 0.007950 | 112 | 148 | 0.303846 | 0.223077 |

## Active Smoke/Inferno Intervals

- `6.5s` - `48.0s`, rows `84`
- `51.0s` - `57.5s`, rows `14`
- `82.0s` - `104.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `93.0`, LSTM `0.6167`, XGBoost `0.7220`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.6294`, XGBoost `0.7219`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.6359`, XGBoost `0.7205`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.2215`, XGBoost `0.3051`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.2225`, XGBoost `0.3051`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.6924`, XGBoost `0.7698`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.6433`, XGBoost `0.7205`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.6472`, XGBoost `0.7219`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6639`, XGBoost `0.7355`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.6515`, XGBoost `0.7219`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
