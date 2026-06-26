# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `9`
- rows: `110`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 110 | 1.000 | 0.821885 | 0.786331 | 0.035554 | 95 | 15 | 1.000000 | 1.000000 |
| active/recent utility | 110 | 1.000 | 0.821885 | 0.786331 | 0.035554 | 95 | 15 | 1.000000 | 1.000000 |
| strong utility action | 108 | 0.982 | 0.824223 | 0.788557 | 0.035665 | 93 | 15 | 1.000000 | 1.000000 |
| utility damage | 22 | 0.200 | 0.810665 | 0.757557 | 0.053107 | 17 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 96 | 0.873 | 0.837846 | 0.800499 | 0.037346 | 82 | 14 | 1.000000 | 1.000000 |
| recent utility last 5s | 15 | 0.136 | 0.726149 | 0.692974 | 0.033174 | 14 | 1 | 1.000000 | 1.000000 |
| flash effect present | 110 | 1.000 | 0.821885 | 0.786331 | 0.035554 | 95 | 15 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `54.5s`, rows `96`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.0`, LSTM `0.7836`, XGBoost `0.6957`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.7592`, XGBoost `0.6712`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.7828`, XGBoost `0.6952`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.7537`, XGBoost `0.6666`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.7860`, XGBoost `0.7000`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `39.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.7535`, XGBoost `0.6712`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.7516`, XGBoost `0.6701`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.7728`, XGBoost `0.6935`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `23.5`, LSTM `0.7796`, XGBoost `0.7007`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `53.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.7789`, XGBoost `0.7001`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `53.0`, recent_utility `0`
