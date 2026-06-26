# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m3-nuke.csv`
- round_num: `14`
- rows: `187`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.588716 | 0.613077 | -0.024361 | 113 | 74 | 0.807487 | 0.823529 |
| active/recent utility | 187 | 1.000 | 0.588716 | 0.613077 | -0.024361 | 113 | 74 | 0.807487 | 0.823529 |
| strong utility action | 177 | 0.947 | 0.580330 | 0.604787 | -0.024457 | 108 | 69 | 0.796610 | 0.813559 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 167 | 0.893 | 0.577126 | 0.602249 | -0.025124 | 107 | 60 | 0.784431 | 0.802395 |
| recent utility last 5s | 10 | 0.053 | 0.633847 | 0.647166 | -0.013319 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 187 | 1.000 | 0.588716 | 0.613077 | -0.024361 | 113 | 74 | 0.807487 | 0.823529 |

## Active Smoke/Inferno Intervals

- `8.0s` - `91.0s`, rows `167`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.5`, LSTM `0.1489`, XGBoost `0.6253`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.1595`, XGBoost `0.6263`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.2013`, XGBoost `0.6267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.2164`, XGBoost `0.6267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2180`, XGBoost `0.6275`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.2196`, XGBoost `0.6275`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1780`, XGBoost `0.5829`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.2265`, XGBoost `0.6277`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.2323`, XGBoost `0.6267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.2332`, XGBoost `0.6267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
