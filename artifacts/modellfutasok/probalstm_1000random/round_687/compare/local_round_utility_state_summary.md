# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `15`
- rows: `248`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 248 | 1.000 | 0.304776 | 0.265086 | 0.039689 | 104 | 144 | 0.560484 | 0.963710 |
| active/recent utility | 248 | 1.000 | 0.304776 | 0.265086 | 0.039689 | 104 | 144 | 0.560484 | 0.963710 |
| strong utility action | 218 | 0.879 | 0.307833 | 0.263121 | 0.044712 | 86 | 132 | 0.532110 | 1.000000 |
| utility damage | 25 | 0.101 | 0.564160 | 0.462248 | 0.101912 | 0 | 25 | 0.000000 | 1.000000 |
| active smoke/inferno | 218 | 0.879 | 0.307833 | 0.263121 | 0.044712 | 86 | 132 | 0.532110 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 248 | 1.000 | 0.304776 | 0.265086 | 0.039689 | 104 | 144 | 0.560484 | 0.963710 |

## Active Smoke/Inferno Intervals

- `8.5s` - `102.0s`, rows `188`
- `104.0s` - `114.0s`, rows `21`
- `119.5s` - `123.5s`, rows `9`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.0`, LSTM `0.6073`, XGBoost `0.4466`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5881`, XGBoost `0.4461`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5913`, XGBoost `0.4495`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5915`, XGBoost `0.4498`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `25.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.4010`, XGBoost `0.2595`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5955`, XGBoost `0.4542`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5724`, XGBoost `0.4461`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5708`, XGBoost `0.4459`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5725`, XGBoost `0.4511`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5662`, XGBoost `0.4459`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
