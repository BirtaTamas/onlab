# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `37`
- rows: `142`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.431559 | 0.456896 | -0.025337 | 114 | 28 | 0.436620 | 0.422535 |
| active/recent utility | 142 | 1.000 | 0.431559 | 0.456896 | -0.025337 | 114 | 28 | 0.436620 | 0.422535 |
| strong utility action | 127 | 0.894 | 0.474138 | 0.499462 | -0.025325 | 99 | 28 | 0.385827 | 0.370079 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 117 | 0.824 | 0.468874 | 0.496886 | -0.028012 | 95 | 22 | 0.418803 | 0.401709 |
| recent utility last 5s | 23 | 0.162 | 0.615671 | 0.616916 | -0.001246 | 14 | 9 | 0.000000 | 0.000000 |
| flash effect present | 142 | 1.000 | 0.431559 | 0.456896 | -0.025337 | 114 | 28 | 0.436620 | 0.422535 |

## Active Smoke/Inferno Intervals

- `6.0s` - `62.0s`, rows `113`
- `69.0s` - `70.5s`, rows `4`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.1029`, XGBoost `0.2255`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.1194`, XGBoost `0.2255`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.1210`, XGBoost `0.2238`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1256`, XGBoost `0.2255`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1217`, XGBoost `0.2188`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.1302`, XGBoost `0.2255`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.1235`, XGBoost `0.2188`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.1289`, XGBoost `0.2238`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.1293`, XGBoost `0.2233`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1321`, XGBoost `0.2238`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
