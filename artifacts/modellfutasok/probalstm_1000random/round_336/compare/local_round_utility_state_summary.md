# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `3`
- rows: `144`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.061330 | 0.157516 | -0.096186 | 144 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 144 | 1.000 | 0.061330 | 0.157516 | -0.096186 | 144 | 0 | 1.000000 | 1.000000 |
| strong utility action | 94 | 0.653 | 0.074491 | 0.172568 | -0.098077 | 94 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 94 | 0.653 | 0.074491 | 0.172568 | -0.098077 | 94 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.061330 | 0.157516 | -0.096186 | 144 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `56.0s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.5`, LSTM `0.0631`, XGBoost `0.2956`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.2656`, XGBoost `0.4936`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0679`, XGBoost `0.2842`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0721`, XGBoost `0.2853`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0697`, XGBoost `0.2776`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1101`, XGBoost `0.3067`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0791`, XGBoost `0.2751`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0779`, XGBoost `0.2723`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1206`, XGBoost `0.3067`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1227`, XGBoost `0.3067`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
