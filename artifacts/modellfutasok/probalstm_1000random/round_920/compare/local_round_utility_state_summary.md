# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `9`
- rows: `274`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 274 | 1.000 | 0.699680 | 0.647032 | 0.052648 | 190 | 84 | 1.000000 | 0.974453 |
| active/recent utility | 274 | 1.000 | 0.699680 | 0.647032 | 0.052648 | 190 | 84 | 1.000000 | 0.974453 |
| strong utility action | 189 | 0.690 | 0.643376 | 0.579368 | 0.064008 | 128 | 61 | 1.000000 | 0.968254 |
| utility damage | 40 | 0.146 | 0.612638 | 0.540115 | 0.072523 | 35 | 5 | 1.000000 | 0.925000 |
| active smoke/inferno | 189 | 0.690 | 0.643376 | 0.579368 | 0.064008 | 128 | 61 | 1.000000 | 0.968254 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 274 | 1.000 | 0.699680 | 0.647032 | 0.052648 | 190 | 84 | 1.000000 | 0.974453 |

## Active Smoke/Inferno Intervals

- `8.0s` - `41.5s`, rows `68`
- `48.0s` - `108.0s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.7065`, XGBoost `0.5078`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7041`, XGBoost `0.5078`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7021`, XGBoost `0.5078`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7015`, XGBoost `0.5078`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.7004`, XGBoost `0.5074`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.7005`, XGBoost `0.5093`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6972`, XGBoost `0.5074`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6974`, XGBoost `0.5078`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6963`, XGBoost `0.5074`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6949`, XGBoost `0.5074`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
