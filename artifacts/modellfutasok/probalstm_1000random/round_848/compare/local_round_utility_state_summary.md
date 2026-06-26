# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `11`
- rows: `172`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 172 | 1.000 | 0.158743 | 0.280107 | -0.121364 | 172 | 0 | 1.000000 | 0.872093 |
| active/recent utility | 172 | 1.000 | 0.158743 | 0.280107 | -0.121364 | 172 | 0 | 1.000000 | 0.872093 |
| strong utility action | 141 | 0.820 | 0.149084 | 0.274840 | -0.125756 | 141 | 0 | 1.000000 | 0.964539 |
| utility damage | 12 | 0.070 | 0.331009 | 0.460086 | -0.129077 | 12 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 141 | 0.820 | 0.149084 | 0.274840 | -0.125756 | 141 | 0 | 1.000000 | 0.964539 |
| recent utility last 5s | 20 | 0.116 | 0.278875 | 0.406752 | -0.127877 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 172 | 1.000 | 0.158743 | 0.280107 | -0.121364 | 172 | 0 | 1.000000 | 0.872093 |

## Active Smoke/Inferno Intervals

- `9.0s` - `73.0s`, rows `129`
- `80.0s` - `85.5s`, rows `12`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.1163`, XGBoost `0.3112`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0695`, XGBoost `0.2644`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1194`, XGBoost `0.3112`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1195`, XGBoost `0.3112`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `1`
- seconds `9.5`, LSTM `0.1142`, XGBoost `0.3056`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.0562`, XGBoost `0.2464`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.3008`, XGBoost `0.4884`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `26.0`, recent_utility `1`
- seconds `25.5`, LSTM `0.2752`, XGBoost `0.4625`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0802`, XGBoost `0.2644`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.2796`, XGBoost `0.4635`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
