# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `3`
- rows: `101`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 101 | 1.000 | 0.019172 | 0.042230 | -0.023058 | 101 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 101 | 1.000 | 0.019172 | 0.042230 | -0.023058 | 101 | 0 | 1.000000 | 1.000000 |
| strong utility action | 45 | 0.446 | 0.028354 | 0.049457 | -0.021103 | 45 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 45 | 0.446 | 0.028354 | 0.049457 | -0.021103 | 45 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 101 | 1.000 | 0.019172 | 0.042230 | -0.023058 | 101 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `31.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.0`, LSTM `0.0297`, XGBoost `0.0863`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0350`, XGBoost `0.0879`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0356`, XGBoost `0.0874`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0375`, XGBoost `0.0865`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0387`, XGBoost `0.0870`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0403`, XGBoost `0.0885`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0401`, XGBoost `0.0847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0430`, XGBoost `0.0856`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0346`, XGBoost `0.0759`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0448`, XGBoost `0.0847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
