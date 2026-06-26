# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `6`
- rows: `100`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 100 | 1.000 | 0.017510 | 0.026458 | -0.008948 | 90 | 10 | 1.000000 | 1.000000 |
| active/recent utility | 100 | 1.000 | 0.017510 | 0.026458 | -0.008948 | 90 | 10 | 1.000000 | 1.000000 |
| strong utility action | 91 | 0.910 | 0.017676 | 0.025964 | -0.008288 | 82 | 9 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 81 | 0.810 | 0.019175 | 0.025265 | -0.006090 | 72 | 9 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.100 | 0.005535 | 0.031625 | -0.026091 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 100 | 1.000 | 0.017510 | 0.026458 | -0.008948 | 90 | 10 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `30.5s`, rows `45`
- `32.0s` - `49.5s`, rows `36`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.0`, LSTM `0.0135`, XGBoost `0.0404`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.0047`, XGBoost `0.0314`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.0048`, XGBoost `0.0314`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.0065`, XGBoost `0.0331`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.0048`, XGBoost `0.0314`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0051`, XGBoost `0.0314`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.0052`, XGBoost `0.0314`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.0057`, XGBoost `0.0316`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0057`, XGBoost `0.0315`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.0068`, XGBoost `0.0323`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
