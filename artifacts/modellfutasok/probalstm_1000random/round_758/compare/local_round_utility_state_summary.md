# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `17`
- rows: `185`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 185 | 1.000 | 0.611834 | 0.628115 | -0.016281 | 96 | 89 | 0.902703 | 0.540541 |
| active/recent utility | 185 | 1.000 | 0.611834 | 0.628115 | -0.016281 | 96 | 89 | 0.902703 | 0.540541 |
| strong utility action | 176 | 0.951 | 0.597753 | 0.615441 | -0.017688 | 94 | 82 | 0.897727 | 0.528409 |
| utility damage | 10 | 0.054 | 0.549951 | 0.476336 | 0.073615 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 160 | 0.865 | 0.602959 | 0.628315 | -0.025356 | 78 | 82 | 0.887500 | 0.556250 |
| recent utility last 5s | 10 | 0.054 | 0.542404 | 0.493276 | 0.049128 | 10 | 0 | 1.000000 | 0.400000 |
| flash effect present | 185 | 1.000 | 0.611834 | 0.628115 | -0.016281 | 96 | 89 | 0.902703 | 0.540541 |

## Active Smoke/Inferno Intervals

- `9.0s` - `88.5s`, rows `160`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.0`, LSTM `0.5782`, XGBoost `0.8132`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.6004`, XGBoost `0.8132`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5484`, XGBoost `0.7428`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5462`, XGBoost `0.7398`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5375`, XGBoost `0.7284`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5361`, XGBoost `0.7255`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.6286`, XGBoost `0.8161`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.4391`, XGBoost `0.6232`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5376`, XGBoost `0.7212`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.6631`, XGBoost `0.8417`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
