# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.804435 | 0.807301 | -0.002866 | 112 | 118 | 0.921739 | 0.982609 |
| active/recent utility | 230 | 1.000 | 0.804435 | 0.807301 | -0.002866 | 112 | 118 | 0.921739 | 0.982609 |
| strong utility action | 167 | 0.726 | 0.800540 | 0.807772 | -0.007233 | 75 | 92 | 0.946108 | 0.976048 |
| utility damage | 31 | 0.135 | 0.758023 | 0.745235 | 0.012788 | 21 | 10 | 0.709677 | 0.870968 |
| active smoke/inferno | 158 | 0.687 | 0.795300 | 0.804948 | -0.009648 | 66 | 92 | 0.943038 | 0.974684 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.804435 | 0.807301 | -0.002866 | 112 | 118 | 0.921739 | 0.982609 |

## Active Smoke/Inferno Intervals

- `6.0s` - `62.5s`, rows `114`
- `73.5s` - `95.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.5678`, XGBoost `0.7304`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.5728`, XGBoost `0.7305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.5974`, XGBoost `0.7475`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5871`, XGBoost `0.7304`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5875`, XGBoost `0.7304`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.5993`, XGBoost `0.7310`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6047`, XGBoost `0.7310`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6378`, XGBoost `0.7273`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4574`, XGBoost `0.5342`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `96.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4314`, XGBoost `0.5030`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `96.0`, recent_utility `0`
