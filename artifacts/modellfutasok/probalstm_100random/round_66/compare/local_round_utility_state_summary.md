# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `11`
- rows: `155`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 155 | 1.000 | 0.434467 | 0.667165 | -0.232698 | 154 | 1 | 0.651613 | 0.400000 |
| active/recent utility | 155 | 1.000 | 0.434467 | 0.667165 | -0.232698 | 154 | 1 | 0.651613 | 0.400000 |
| strong utility action | 99 | 0.639 | 0.446668 | 0.629172 | -0.182504 | 98 | 1 | 0.636364 | 0.545455 |
| utility damage | 10 | 0.065 | 0.406003 | 0.464493 | -0.058490 | 9 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 90 | 0.581 | 0.461149 | 0.648506 | -0.187357 | 89 | 1 | 0.600000 | 0.500000 |
| recent utility last 5s | 9 | 0.058 | 0.301860 | 0.435835 | -0.133975 | 9 | 0 | 1.000000 | 1.000000 |
| flash effect present | 155 | 1.000 | 0.434467 | 0.667165 | -0.232698 | 154 | 1 | 0.651613 | 0.400000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `55.0s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.4415`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4526`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.4552`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.4627`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4725`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.4732`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4739`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.4760`, XGBoost `0.8610`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4983`, XGBoost `0.8582`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5130`, XGBoost `0.8578`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
