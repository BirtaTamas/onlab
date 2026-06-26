# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `11`
- rows: `201`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 201 | 1.000 | 0.685188 | 0.636883 | 0.048305 | 156 | 45 | 1.000000 | 0.980100 |
| active/recent utility | 201 | 1.000 | 0.685188 | 0.636883 | 0.048305 | 156 | 45 | 1.000000 | 0.980100 |
| strong utility action | 165 | 0.821 | 0.673955 | 0.616631 | 0.057324 | 137 | 28 | 1.000000 | 0.975758 |
| utility damage | 43 | 0.214 | 0.680932 | 0.636372 | 0.044560 | 30 | 13 | 1.000000 | 0.976744 |
| active smoke/inferno | 162 | 0.806 | 0.675611 | 0.617219 | 0.058392 | 135 | 27 | 1.000000 | 0.975309 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 201 | 1.000 | 0.685188 | 0.636883 | 0.048305 | 156 | 45 | 1.000000 | 0.980100 |

## Active Smoke/Inferno Intervals

- `11.0s` - `18.0s`, rows `15`
- `20.0s` - `25.0s`, rows `11`
- `26.0s` - `93.5s`, rows `136`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.5`, LSTM `0.7720`, XGBoost `0.5229`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `98.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.7518`, XGBoost `0.5073`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `77.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.7409`, XGBoost `0.5011`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.7181`, XGBoost `0.4943`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.7221`, XGBoost `0.5123`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `114.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.7222`, XGBoost `0.5181`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `114.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.7430`, XGBoost `0.5420`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.7180`, XGBoost `0.5229`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `114.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.6460`, XGBoost `0.4548`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.6004`, XGBoost `0.4163`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
