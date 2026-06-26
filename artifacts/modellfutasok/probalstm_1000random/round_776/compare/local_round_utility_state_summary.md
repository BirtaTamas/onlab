# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `7`
- rows: `108`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 108 | 1.000 | 0.008942 | 0.035747 | -0.026805 | 108 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 108 | 1.000 | 0.008942 | 0.035747 | -0.026805 | 108 | 0 | 1.000000 | 1.000000 |
| strong utility action | 60 | 0.556 | 0.010033 | 0.037516 | -0.027483 | 60 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 60 | 0.556 | 0.010033 | 0.037516 | -0.027483 | 60 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 108 | 1.000 | 0.008942 | 0.035747 | -0.026805 | 108 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `30.0s`, rows `46`
- `32.5s` - `39.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.0171`, XGBoost `0.1036`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0177`, XGBoost `0.1034`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0193`, XGBoost `0.1047`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0194`, XGBoost `0.1041`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0180`, XGBoost `0.1025`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0218`, XGBoost `0.1039`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0202`, XGBoost `0.1021`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0230`, XGBoost `0.1040`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0234`, XGBoost `0.1040`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0227`, XGBoost `0.1024`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
