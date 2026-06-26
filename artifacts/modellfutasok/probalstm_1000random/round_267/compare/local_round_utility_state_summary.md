# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `9`
- rows: `283`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 283 | 1.000 | 0.252832 | 0.343776 | -0.090945 | 283 | 0 | 1.000000 | 0.484099 |
| active/recent utility | 283 | 1.000 | 0.252832 | 0.343776 | -0.090945 | 283 | 0 | 1.000000 | 0.484099 |
| strong utility action | 186 | 0.657 | 0.295462 | 0.402672 | -0.107210 | 186 | 0 | 1.000000 | 0.349462 |
| utility damage | 10 | 0.035 | 0.406365 | 0.508231 | -0.101866 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 186 | 0.657 | 0.295462 | 0.402672 | -0.107210 | 186 | 0 | 1.000000 | 0.349462 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 283 | 1.000 | 0.252832 | 0.343776 | -0.090945 | 283 | 0 | 1.000000 | 0.484099 |

## Active Smoke/Inferno Intervals

- `10.5s` - `65.0s`, rows `110`
- `74.5s` - `112.0s`, rows `76`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `84.5`, LSTM `0.2624`, XGBoost `0.5113`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.2612`, XGBoost `0.5073`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.2806`, XGBoost `0.5073`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.2606`, XGBoost `0.4723`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.2978`, XGBoost `0.5073`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3013`, XGBoost `0.5103`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.2937`, XGBoost `0.5012`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.3080`, XGBoost `0.5093`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3099`, XGBoost `0.5103`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.3101`, XGBoost `0.5103`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
