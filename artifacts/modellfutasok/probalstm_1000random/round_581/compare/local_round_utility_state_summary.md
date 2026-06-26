# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `12`
- rows: `100`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 100 | 1.000 | 0.228805 | 0.327125 | -0.098319 | 87 | 13 | 0.940000 | 0.640000 |
| active/recent utility | 100 | 1.000 | 0.228805 | 0.327125 | -0.098319 | 87 | 13 | 0.940000 | 0.640000 |
| strong utility action | 97 | 0.970 | 0.222538 | 0.318934 | -0.096396 | 84 | 13 | 0.938144 | 0.659794 |
| utility damage | 31 | 0.310 | 0.271967 | 0.352329 | -0.080362 | 27 | 4 | 0.806452 | 0.677419 |
| active smoke/inferno | 88 | 0.880 | 0.214164 | 0.291460 | -0.077295 | 75 | 13 | 0.931818 | 0.727273 |
| recent utility last 5s | 10 | 0.100 | 0.302705 | 0.588376 | -0.285670 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 100 | 1.000 | 0.228805 | 0.327125 | -0.098319 | 87 | 13 | 0.940000 | 0.640000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `49.5s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `5.0`, LSTM `0.2662`, XGBoost `0.5875`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.2758`, XGBoost `0.5905`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.2821`, XGBoost `0.5905`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.2874`, XGBoost `0.5957`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.2820`, XGBoost `0.5875`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.2740`, XGBoost `0.5777`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.2854`, XGBoost `0.5815`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.3147`, XGBoost `0.5905`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `23.5`, LSTM `0.0641`, XGBoost `0.3255`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0594`, XGBoost `0.3206`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
