# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `10`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.774964 | 0.770083 | 0.004881 | 63 | 81 | 1.000000 | 1.000000 |
| active/recent utility | 144 | 1.000 | 0.774964 | 0.770083 | 0.004881 | 63 | 81 | 1.000000 | 1.000000 |
| strong utility action | 105 | 0.729 | 0.725633 | 0.712023 | 0.013610 | 61 | 44 | 1.000000 | 1.000000 |
| utility damage | 25 | 0.174 | 0.643398 | 0.567839 | 0.075559 | 25 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 95 | 0.660 | 0.747106 | 0.727655 | 0.019450 | 61 | 34 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.069 | 0.521642 | 0.563519 | -0.041877 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 144 | 1.000 | 0.774964 | 0.770083 | 0.004881 | 63 | 81 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `54.0s`, rows `95`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.5`, LSTM `0.6653`, XGBoost `0.5668`, closer `lstm`, smoke `2`, inferno `4`, utility_damage `50.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6623`, XGBoost `0.5668`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `49.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6083`, XGBoost `0.5142`, closer `lstm`, smoke `2`, inferno `5`, utility_damage `28.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6828`, XGBoost `0.5902`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6814`, XGBoost `0.5912`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6744`, XGBoost `0.5868`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6321`, XGBoost `0.5491`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `18.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6674`, XGBoost `0.5846`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6670`, XGBoost `0.5846`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `53.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6455`, XGBoost `0.5661`, closer `lstm`, smoke `2`, inferno `5`, utility_damage `49.0`, recent_utility `0`
