# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `18`
- rows: `142`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.846359 | 0.829226 | 0.017132 | 109 | 33 | 1.000000 | 1.000000 |
| active/recent utility | 142 | 1.000 | 0.846359 | 0.829226 | 0.017132 | 109 | 33 | 1.000000 | 1.000000 |
| strong utility action | 96 | 0.676 | 0.816226 | 0.795434 | 0.020793 | 77 | 19 | 1.000000 | 1.000000 |
| utility damage | 16 | 0.113 | 0.765257 | 0.740448 | 0.024810 | 16 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 86 | 0.606 | 0.815010 | 0.800214 | 0.014796 | 67 | 19 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.070 | 0.826684 | 0.754323 | 0.072361 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 142 | 1.000 | 0.846359 | 0.829226 | 0.017132 | 109 | 33 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `49.0s`, rows `86`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.5`, LSTM `0.8293`, XGBoost `0.7420`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.8291`, XGBoost `0.7428`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.8294`, XGBoost `0.7491`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.8293`, XGBoost `0.7496`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.8239`, XGBoost `0.7463`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.8259`, XGBoost `0.7496`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `24.0`, LSTM `0.8082`, XGBoost `0.7424`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.8249`, XGBoost `0.7601`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.8293`, XGBoost `0.7679`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.8270`, XGBoost `0.7679`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
