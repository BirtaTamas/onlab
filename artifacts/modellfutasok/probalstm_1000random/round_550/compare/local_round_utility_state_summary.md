# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `11`
- rows: `185`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 185 | 1.000 | 0.344667 | 0.542765 | -0.198098 | 0 | 185 | 0.102703 | 0.459459 |
| active/recent utility | 185 | 1.000 | 0.344667 | 0.542765 | -0.198098 | 0 | 185 | 0.102703 | 0.459459 |
| strong utility action | 149 | 0.805 | 0.337752 | 0.529657 | -0.191905 | 0 | 149 | 0.080537 | 0.362416 |
| utility damage | 10 | 0.054 | 0.246141 | 0.491516 | -0.245376 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 138 | 0.746 | 0.331466 | 0.533250 | -0.201784 | 0 | 138 | 0.086957 | 0.391304 |
| recent utility last 5s | 11 | 0.059 | 0.416607 | 0.484585 | -0.067978 | 0 | 11 | 0.000000 | 0.000000 |
| flash effect present | 185 | 1.000 | 0.344667 | 0.542765 | -0.198098 | 0 | 185 | 0.102703 | 0.459459 |

## Active Smoke/Inferno Intervals

- `6.5s` - `37.0s`, rows `62`
- `51.0s` - `88.5s`, rows `76`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.3900`, XGBoost `0.7344`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.1660`, XGBoost `0.5100`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.1755`, XGBoost `0.5105`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.1929`, XGBoost `0.5105`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1938`, XGBoost `0.5102`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.2002`, XGBoost `0.5089`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.2103`, XGBoost `0.5081`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.2039`, XGBoost `0.4919`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2167`, XGBoost `0.5042`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.2195`, XGBoost `0.5044`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
