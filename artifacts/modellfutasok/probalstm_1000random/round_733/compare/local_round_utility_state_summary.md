# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `7`
- rows: `116`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.112209 | 0.240151 | -0.127943 | 116 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 116 | 1.000 | 0.112209 | 0.240151 | -0.127943 | 116 | 0 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.922 | 0.117263 | 0.251313 | -0.134049 | 107 | 0 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.095 | 0.332178 | 0.370879 | -0.038701 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 87 | 0.750 | 0.120847 | 0.264620 | -0.143773 | 87 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.172 | 0.101674 | 0.193425 | -0.091751 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 116 | 1.000 | 0.112209 | 0.240151 | -0.127943 | 116 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `49.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.0`, LSTM `0.0580`, XGBoost `0.3072`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0474`, XGBoost `0.2954`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0396`, XGBoost `0.2862`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0631`, XGBoost `0.3087`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0525`, XGBoost `0.2896`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0545`, XGBoost `0.2890`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0759`, XGBoost `0.3025`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `44.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0315`, XGBoost `0.2574`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0832`, XGBoost `0.3090`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0354`, XGBoost `0.2585`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
