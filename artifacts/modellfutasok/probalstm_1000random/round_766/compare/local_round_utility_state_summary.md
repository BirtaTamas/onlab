# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `15`
- rows: `164`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.570604 | 0.436277 | 0.134327 | 161 | 3 | 0.957317 | 0.030488 |
| active/recent utility | 164 | 1.000 | 0.570604 | 0.436277 | 0.134327 | 161 | 3 | 0.957317 | 0.030488 |
| strong utility action | 135 | 0.823 | 0.573721 | 0.431048 | 0.142673 | 132 | 3 | 0.948148 | 0.029630 |
| utility damage | 38 | 0.232 | 0.583359 | 0.455822 | 0.127537 | 35 | 3 | 0.868421 | 0.105263 |
| active smoke/inferno | 125 | 0.762 | 0.576567 | 0.432430 | 0.144137 | 122 | 3 | 0.944000 | 0.032000 |
| recent utility last 5s | 11 | 0.067 | 0.538315 | 0.413541 | 0.124773 | 11 | 0 | 1.000000 | 0.000000 |
| flash effect present | 164 | 1.000 | 0.570604 | 0.436277 | 0.134327 | 161 | 3 | 0.957317 | 0.030488 |

## Active Smoke/Inferno Intervals

- `9.0s` - `68.5s`, rows `120`
- `79.5s` - `81.5s`, rows `5`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.0`, LSTM `0.4897`, XGBoost `0.2001`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6290`, XGBoost `0.4167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.6375`, XGBoost `0.4264`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6272`, XGBoost `0.4167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6197`, XGBoost `0.4102`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6223`, XGBoost `0.4167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6185`, XGBoost `0.4153`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6182`, XGBoost `0.4167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.6176`, XGBoost `0.4167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6091`, XGBoost `0.4099`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `1.0`, recent_utility `0`
