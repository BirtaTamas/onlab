# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `12`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.595432 | 0.715923 | -0.120491 | 0 | 230 | 0.860870 | 0.886957 |
| active/recent utility | 230 | 1.000 | 0.595432 | 0.715923 | -0.120491 | 0 | 230 | 0.860870 | 0.886957 |
| strong utility action | 207 | 0.900 | 0.610312 | 0.733542 | -0.123230 | 0 | 207 | 0.922705 | 0.932367 |
| utility damage | 20 | 0.087 | 0.726654 | 0.810334 | -0.083680 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 207 | 0.900 | 0.610312 | 0.733542 | -0.123230 | 0 | 207 | 0.922705 | 0.932367 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.595432 | 0.715923 | -0.120491 | 0 | 230 | 0.860870 | 0.886957 |

## Active Smoke/Inferno Intervals

- `8.0s` - `70.5s`, rows `126`
- `74.0s` - `95.5s`, rows `44`
- `96.5s` - `114.5s`, rows `37`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.0`, LSTM `0.5306`, XGBoost `0.7303`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.5351`, XGBoost `0.7236`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.5441`, XGBoost `0.7322`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.5398`, XGBoost `0.7236`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.5411`, XGBoost `0.7236`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.5413`, XGBoost `0.7235`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.5413`, XGBoost `0.7235`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.5503`, XGBoost `0.7322`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.5499`, XGBoost `0.7296`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.5434`, XGBoost `0.7224`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
