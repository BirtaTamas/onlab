# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `12`
- rows: `158`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.440306 | 0.505964 | -0.065658 | 146 | 12 | 0.487342 | 0.405063 |
| active/recent utility | 158 | 1.000 | 0.440306 | 0.505964 | -0.065658 | 146 | 12 | 0.487342 | 0.405063 |
| strong utility action | 142 | 0.899 | 0.429358 | 0.499049 | -0.069691 | 131 | 11 | 0.542254 | 0.450704 |
| utility damage | 25 | 0.158 | 0.512850 | 0.591926 | -0.079076 | 21 | 4 | 0.360000 | 0.160000 |
| active smoke/inferno | 142 | 0.899 | 0.429358 | 0.499049 | -0.069691 | 131 | 11 | 0.542254 | 0.450704 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.440306 | 0.505964 | -0.065658 | 146 | 12 | 0.487342 | 0.405063 |

## Active Smoke/Inferno Intervals

- `8.0s` - `78.5s`, rows `142`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.2969`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.3053`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3054`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.3181`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.3774`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.4110`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.4373`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.6779`, XGBoost `0.8489`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `83.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4616`, XGBoost `0.6212`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7026`, XGBoost `0.8489`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `83.0`, recent_utility `0`
