# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `2`
- rows: `134`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.113030 | 0.132448 | -0.019418 | 126 | 8 | 1.000000 | 1.000000 |
| active/recent utility | 134 | 1.000 | 0.113030 | 0.132448 | -0.019418 | 126 | 8 | 1.000000 | 1.000000 |
| strong utility action | 93 | 0.694 | 0.097408 | 0.114979 | -0.017571 | 86 | 7 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 93 | 0.694 | 0.097408 | 0.114979 | -0.017571 | 86 | 7 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 134 | 1.000 | 0.113030 | 0.132448 | -0.019418 | 126 | 8 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `56.0s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.3042`, XGBoost `0.4050`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.3150`, XGBoost `0.4043`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4115`, XGBoost `0.3361`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4159`, XGBoost `0.3430`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0537`, XGBoost `0.1219`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0553`, XGBoost `0.1229`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4129`, XGBoost `0.3467`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0567`, XGBoost `0.1226`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0600`, XGBoost `0.1236`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0606`, XGBoost `0.1226`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `19.0`, recent_utility `0`
