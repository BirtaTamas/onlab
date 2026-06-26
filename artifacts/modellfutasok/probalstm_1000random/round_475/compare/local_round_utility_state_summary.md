# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `7`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.829134 | 0.841734 | -0.012600 | 91 | 139 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.829134 | 0.841734 | -0.012600 | 91 | 139 | 1.000000 | 1.000000 |
| strong utility action | 171 | 0.743 | 0.840431 | 0.867847 | -0.027415 | 32 | 139 | 1.000000 | 1.000000 |
| utility damage | 14 | 0.061 | 0.731719 | 0.737430 | -0.005711 | 6 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 171 | 0.743 | 0.840431 | 0.867847 | -0.027415 | 32 | 139 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.829134 | 0.841734 | -0.012600 | 91 | 139 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `95.5s`, rows `171`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.8171`, XGBoost `0.8870`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.8208`, XGBoost `0.8871`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.8215`, XGBoost `0.8854`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.8236`, XGBoost `0.8870`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.8240`, XGBoost `0.8854`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6967`, XGBoost `0.7576`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `1.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.8264`, XGBoost `0.8870`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.8255`, XGBoost `0.8849`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6030`, XGBoost `0.5439`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.8275`, XGBoost `0.8854`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
