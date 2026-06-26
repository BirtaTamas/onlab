# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `18`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.659329 | 0.614749 | 0.044580 | 76 | 59 | 0.851852 | 0.829630 |
| active/recent utility | 135 | 1.000 | 0.659329 | 0.614749 | 0.044580 | 76 | 59 | 0.851852 | 0.829630 |
| strong utility action | 86 | 0.637 | 0.558207 | 0.510369 | 0.047838 | 50 | 36 | 0.767442 | 0.732558 |
| utility damage | 10 | 0.074 | 0.538499 | 0.485635 | 0.052864 | 5 | 5 | 0.700000 | 0.700000 |
| active smoke/inferno | 86 | 0.637 | 0.558207 | 0.510369 | 0.047838 | 50 | 36 | 0.767442 | 0.732558 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.659329 | 0.614749 | 0.044580 | 76 | 59 | 0.851852 | 0.829630 |

## Active Smoke/Inferno Intervals

- `4.0s` - `46.5s`, rows `86`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.0`, LSTM `0.5124`, XGBoost `0.2637`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.4903`, XGBoost `0.2637`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.4859`, XGBoost `0.2637`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.4906`, XGBoost `0.2722`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4806`, XGBoost `0.2637`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.4765`, XGBoost `0.2637`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.4811`, XGBoost `0.2722`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.4709`, XGBoost `0.2696`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.4718`, XGBoost `0.2722`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.4636`, XGBoost `0.2722`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
