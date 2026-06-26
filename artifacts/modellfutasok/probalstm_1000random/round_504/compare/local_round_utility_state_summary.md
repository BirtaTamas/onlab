# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m3-ancient.csv`
- round_num: `9`
- rows: `215`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.359145 | 0.315565 | 0.043580 | 61 | 154 | 0.744186 | 0.781395 |
| active/recent utility | 215 | 1.000 | 0.359145 | 0.315565 | 0.043580 | 61 | 154 | 0.744186 | 0.781395 |
| strong utility action | 122 | 0.567 | 0.488568 | 0.413350 | 0.075218 | 15 | 107 | 0.565574 | 0.631148 |
| utility damage | 22 | 0.102 | 0.540652 | 0.449292 | 0.091359 | 0 | 22 | 0.454545 | 0.454545 |
| active smoke/inferno | 112 | 0.521 | 0.469050 | 0.397431 | 0.071618 | 15 | 97 | 0.616071 | 0.687500 |
| recent utility last 5s | 10 | 0.047 | 0.707175 | 0.591644 | 0.115531 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 215 | 1.000 | 0.359145 | 0.315565 | 0.043580 | 61 | 154 | 0.744186 | 0.781395 |

## Active Smoke/Inferno Intervals

- `6.0s` - `36.5s`, rows `62`
- `73.5s` - `98.0s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.5`, LSTM `0.4675`, XGBoost `0.2128`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.4610`, XGBoost `0.2096`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.3912`, XGBoost `0.2100`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.3891`, XGBoost `0.2097`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.3740`, XGBoost `0.1961`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.3635`, XGBoost `0.1957`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5469`, XGBoost `0.3898`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5458`, XGBoost `0.3939`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5401`, XGBoost `0.3899`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5455`, XGBoost `0.3955`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
