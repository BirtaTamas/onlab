# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `1`
- rows: `146`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.649515 | 0.666500 | -0.016985 | 68 | 78 | 1.000000 | 0.904110 |
| active/recent utility | 146 | 1.000 | 0.649515 | 0.666500 | -0.016985 | 68 | 78 | 1.000000 | 0.904110 |
| strong utility action | 44 | 0.301 | 0.721901 | 0.773421 | -0.051519 | 10 | 34 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.301 | 0.721901 | 0.773421 | -0.051519 | 10 | 34 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 146 | 1.000 | 0.649515 | 0.666500 | -0.016985 | 68 | 78 | 1.000000 | 0.904110 |

## Active Smoke/Inferno Intervals

- `39.0s` - `60.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.5858`, XGBoost `0.7757`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.5991`, XGBoost `0.7732`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6034`, XGBoost `0.7732`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.5964`, XGBoost `0.7651`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.6281`, XGBoost `0.7756`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6422`, XGBoost `0.7715`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6541`, XGBoost `0.7598`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.6701`, XGBoost `0.7701`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6965`, XGBoost `0.7896`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7269`, XGBoost `0.8147`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
