# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `22`
- rows: `238`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 238 | 1.000 | 0.653372 | 0.679722 | -0.026350 | 63 | 175 | 0.621849 | 0.936975 |
| active/recent utility | 238 | 1.000 | 0.653372 | 0.679722 | -0.026350 | 63 | 175 | 0.621849 | 0.936975 |
| strong utility action | 191 | 0.803 | 0.643396 | 0.663347 | -0.019952 | 63 | 128 | 0.638743 | 0.921466 |
| utility damage | 37 | 0.155 | 0.573313 | 0.600288 | -0.026975 | 10 | 27 | 0.567568 | 1.000000 |
| active smoke/inferno | 182 | 0.765 | 0.650080 | 0.667976 | -0.017896 | 63 | 119 | 0.642857 | 0.917582 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 238 | 1.000 | 0.653372 | 0.679722 | -0.026350 | 63 | 175 | 0.621849 | 0.936975 |

## Active Smoke/Inferno Intervals

- `10.5s` - `17.5s`, rows `15`
- `18.5s` - `25.0s`, rows `14`
- `29.5s` - `76.0s`, rows `94`
- `89.5s` - `118.5s`, rows `59`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `93.0`, LSTM `0.7418`, XGBoost `0.8537`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.4782`, XGBoost `0.3664`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.8618`, XGBoost `0.7501`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.4757`, XGBoost `0.3664`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.8660`, XGBoost `0.7580`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.7281`, XGBoost `0.8351`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.7488`, XGBoost `0.8533`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.7475`, XGBoost `0.8511`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7461`, XGBoost `0.8486`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.7432`, XGBoost `0.8409`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
