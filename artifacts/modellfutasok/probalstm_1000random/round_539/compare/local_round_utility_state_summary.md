# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `10`
- rows: `176`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.169828 | 0.157859 | 0.011968 | 82 | 94 | 0.818182 | 0.761364 |
| active/recent utility | 176 | 1.000 | 0.169828 | 0.157859 | 0.011968 | 82 | 94 | 0.818182 | 0.761364 |
| strong utility action | 128 | 0.727 | 0.153445 | 0.135875 | 0.017570 | 50 | 78 | 0.828125 | 0.828125 |
| utility damage | 10 | 0.057 | 0.285910 | 0.307713 | -0.021803 | 7 | 3 | 0.900000 | 0.900000 |
| active smoke/inferno | 128 | 0.727 | 0.153445 | 0.135875 | 0.017570 | 50 | 78 | 0.828125 | 0.828125 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 176 | 1.000 | 0.169828 | 0.157859 | 0.011968 | 82 | 94 | 0.818182 | 0.761364 |

## Active Smoke/Inferno Intervals

- `10.0s` - `55.0s`, rows `91`
- `69.5s` - `87.5s`, rows `37`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.3952`, XGBoost `0.2421`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3120`, XGBoost `0.1835`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3040`, XGBoost `0.1819`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2795`, XGBoost `0.1585`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3621`, XGBoost `0.2536`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6074`, XGBoost `0.5023`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6109`, XGBoost `0.5064`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6144`, XGBoost `0.5100`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5942`, XGBoost `0.5014`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5990`, XGBoost `0.5098`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `3.0`, recent_utility `0`
