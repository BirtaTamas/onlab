# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `34`
- rows: `265`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 265 | 1.000 | 0.464003 | 0.418144 | 0.045859 | 49 | 216 | 0.301887 | 0.441509 |
| active/recent utility | 265 | 1.000 | 0.464003 | 0.418144 | 0.045859 | 49 | 216 | 0.301887 | 0.441509 |
| strong utility action | 246 | 0.928 | 0.468652 | 0.426204 | 0.042448 | 49 | 197 | 0.292683 | 0.410569 |
| utility damage | 39 | 0.147 | 0.520464 | 0.469332 | 0.051131 | 0 | 39 | 0.205128 | 0.205128 |
| active smoke/inferno | 236 | 0.891 | 0.464245 | 0.423060 | 0.041185 | 49 | 187 | 0.305085 | 0.411017 |
| recent utility last 5s | 10 | 0.038 | 0.572651 | 0.500393 | 0.072259 | 0 | 10 | 0.000000 | 0.400000 |
| flash effect present | 265 | 1.000 | 0.464003 | 0.418144 | 0.045859 | 49 | 216 | 0.301887 | 0.441509 |

## Active Smoke/Inferno Intervals

- `10.5s` - `123.5s`, rows `227`
- `128.0s` - `132.0s`, rows `9`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.0`, LSTM `0.4795`, XGBoost `0.2609`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.4419`, XGBoost `0.2276`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.4598`, XGBoost `0.2606`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.4337`, XGBoost `0.2606`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.5498`, XGBoost `0.3885`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `128.0`, LSTM `0.2163`, XGBoost `0.0600`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `118.5`, LSTM `0.2218`, XGBoost `0.0675`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.5367`, XGBoost `0.3870`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.4354`, XGBoost `0.2863`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `119.0`, LSTM `0.2090`, XGBoost `0.0675`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
