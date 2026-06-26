# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `11`
- rows: `109`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 109 | 1.000 | 0.765784 | 0.723446 | 0.042339 | 77 | 32 | 1.000000 | 1.000000 |
| active/recent utility | 109 | 1.000 | 0.765784 | 0.723446 | 0.042339 | 77 | 32 | 1.000000 | 1.000000 |
| strong utility action | 92 | 0.844 | 0.786777 | 0.753013 | 0.033764 | 60 | 32 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.092 | 0.664100 | 0.545327 | 0.118773 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 92 | 0.844 | 0.786777 | 0.753013 | 0.033764 | 60 | 32 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 109 | 1.000 | 0.765784 | 0.723446 | 0.042339 | 77 | 32 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `54.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.0`, LSTM `0.6888`, XGBoost `0.5400`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6859`, XGBoost `0.5400`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6848`, XGBoost `0.5428`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6790`, XGBoost `0.5400`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6753`, XGBoost `0.5400`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6831`, XGBoost `0.5484`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6725`, XGBoost `0.5402`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6782`, XGBoost `0.5473`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6657`, XGBoost `0.5405`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6652`, XGBoost `0.5402`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
