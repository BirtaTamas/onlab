# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `5`
- rows: `269`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 269 | 1.000 | 0.426593 | 0.478307 | -0.051714 | 44 | 225 | 0.371747 | 0.546468 |
| active/recent utility | 269 | 1.000 | 0.426593 | 0.478307 | -0.051714 | 44 | 225 | 0.371747 | 0.546468 |
| strong utility action | 154 | 0.572 | 0.462146 | 0.489552 | -0.027406 | 39 | 115 | 0.519481 | 0.688312 |
| utility damage | 31 | 0.115 | 0.524969 | 0.560581 | -0.035611 | 5 | 26 | 0.709677 | 1.000000 |
| active smoke/inferno | 145 | 0.539 | 0.454837 | 0.486231 | -0.031394 | 30 | 115 | 0.489655 | 0.668966 |
| recent utility last 5s | 20 | 0.074 | 0.517299 | 0.532731 | -0.015432 | 11 | 9 | 0.550000 | 1.000000 |
| flash effect present | 269 | 1.000 | 0.426593 | 0.478307 | -0.051714 | 44 | 225 | 0.371747 | 0.546468 |

## Active Smoke/Inferno Intervals

- `6.0s` - `45.0s`, rows `79`
- `73.5s` - `106.0s`, rows `66`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.0`, LSTM `0.3241`, XGBoost `0.1517`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.5107`, XGBoost `0.6307`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4087`, XGBoost `0.5257`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `13.5`, LSTM `0.4453`, XGBoost `0.5539`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `31.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.5232`, XGBoost `0.6318`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.4499`, XGBoost `0.5569`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `28.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.3862`, XGBoost `0.4915`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4557`, XGBoost `0.5569`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `33.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4607`, XGBoost `0.5569`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `33.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.4061`, XGBoost `0.4991`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
