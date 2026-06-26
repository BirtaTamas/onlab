# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `12`
- rows: `205`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.171573 | 0.299919 | -0.128346 | 191 | 14 | 1.000000 | 0.975610 |
| active/recent utility | 205 | 1.000 | 0.171573 | 0.299919 | -0.128346 | 191 | 14 | 1.000000 | 0.975610 |
| strong utility action | 184 | 0.898 | 0.169897 | 0.302100 | -0.132203 | 176 | 8 | 1.000000 | 0.972826 |
| utility damage | 22 | 0.107 | 0.220849 | 0.402443 | -0.181595 | 22 | 0 | 1.000000 | 0.772727 |
| active smoke/inferno | 184 | 0.898 | 0.169897 | 0.302100 | -0.132203 | 176 | 8 | 1.000000 | 0.972826 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 205 | 1.000 | 0.171573 | 0.299919 | -0.128346 | 191 | 14 | 1.000000 | 0.975610 |

## Active Smoke/Inferno Intervals

- `3.0s` - `44.0s`, rows `83`
- `46.0s` - `67.5s`, rows `44`
- `73.5s` - `95.0s`, rows `44`
- `96.0s` - `102.0s`, rows `13`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.1047`, XGBoost `0.3916`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1622`, XGBoost `0.4075`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1002`, XGBoost `0.3407`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1025`, XGBoost `0.3375`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1062`, XGBoost `0.3407`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1005`, XGBoost `0.3347`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1077`, XGBoost `0.3347`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1143`, XGBoost `0.3388`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1192`, XGBoost `0.3437`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1107`, XGBoost `0.3347`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
