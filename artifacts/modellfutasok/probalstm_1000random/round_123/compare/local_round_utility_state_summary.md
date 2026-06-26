# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `12`
- rows: `182`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.677482 | 0.693019 | -0.015536 | 94 | 88 | 1.000000 | 1.000000 |
| active/recent utility | 182 | 1.000 | 0.677482 | 0.693019 | -0.015536 | 94 | 88 | 1.000000 | 1.000000 |
| strong utility action | 169 | 0.929 | 0.688761 | 0.705945 | -0.017183 | 84 | 85 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.110 | 0.550389 | 0.520931 | 0.029458 | 20 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 162 | 0.890 | 0.694386 | 0.713631 | -0.019246 | 77 | 85 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 182 | 1.000 | 0.677482 | 0.693019 | -0.015536 | 94 | 88 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `36.0s`, rows `61`
- `40.5s` - `90.5s`, rows `101`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.5`, LSTM `0.6885`, XGBoost `0.8725`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `39.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.7166`, XGBoost `0.8753`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.7156`, XGBoost `0.8675`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `39.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.7320`, XGBoost `0.8742`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.7425`, XGBoost `0.8809`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `39.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.7193`, XGBoost `0.8416`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.7696`, XGBoost `0.8848`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `33.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.7300`, XGBoost `0.8416`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.7306`, XGBoost `0.8417`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.7675`, XGBoost `0.8769`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `48.0`, recent_utility `0`
