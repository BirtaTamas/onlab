# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `8`
- rows: `214`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.567806 | 0.550047 | 0.017759 | 111 | 103 | 0.724299 | 0.785047 |
| active/recent utility | 214 | 1.000 | 0.567806 | 0.550047 | 0.017759 | 111 | 103 | 0.724299 | 0.785047 |
| strong utility action | 187 | 0.874 | 0.558107 | 0.535617 | 0.022490 | 108 | 79 | 0.770053 | 0.770053 |
| utility damage | 10 | 0.047 | 0.833172 | 0.882349 | -0.049177 | 2 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 186 | 0.869 | 0.555849 | 0.533148 | 0.022701 | 108 | 78 | 0.768817 | 0.768817 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.567806 | 0.550047 | 0.017759 | 111 | 103 | 0.724299 | 0.785047 |

## Active Smoke/Inferno Intervals

- `8.0s` - `64.0s`, rows `113`
- `66.0s` - `102.0s`, rows `73`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.5273`, XGBoost `0.2968`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.5122`, XGBoost `0.2833`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.7907`, XGBoost `0.5679`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.5067`, XGBoost `0.2914`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.5177`, XGBoost `0.3029`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.5022`, XGBoost `0.2914`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.7493`, XGBoost `0.5386`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.7773`, XGBoost `0.5679`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.5039`, XGBoost `0.2960`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.7733`, XGBoost `0.5679`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
