# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `9`
- rows: `198`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.367673 | 0.376023 | -0.008350 | 127 | 71 | 0.479798 | 0.479798 |
| active/recent utility | 198 | 1.000 | 0.367673 | 0.376023 | -0.008350 | 127 | 71 | 0.479798 | 0.479798 |
| strong utility action | 142 | 0.717 | 0.439925 | 0.449826 | -0.009901 | 87 | 55 | 0.387324 | 0.387324 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 142 | 0.717 | 0.439925 | 0.449826 | -0.009901 | 87 | 55 | 0.387324 | 0.387324 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.367673 | 0.376023 | -0.008350 | 127 | 71 | 0.479798 | 0.479798 |

## Active Smoke/Inferno Intervals

- `8.0s` - `78.5s`, rows `142`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.4030`, XGBoost `0.2512`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1491`, XGBoost `0.2559`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6421`, XGBoost `0.5424`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6483`, XGBoost `0.5490`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6357`, XGBoost `0.5424`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6313`, XGBoost `0.5398`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6319`, XGBoost `0.5419`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6682`, XGBoost `0.5791`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6296`, XGBoost `0.5419`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6272`, XGBoost `0.5419`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
