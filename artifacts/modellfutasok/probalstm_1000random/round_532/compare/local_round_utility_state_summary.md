# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `4`
- rows: `194`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.136726 | 0.178920 | -0.042193 | 194 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 194 | 1.000 | 0.136726 | 0.178920 | -0.042193 | 194 | 0 | 1.000000 | 1.000000 |
| strong utility action | 126 | 0.649 | 0.164700 | 0.222457 | -0.057757 | 126 | 0 | 1.000000 | 1.000000 |
| utility damage | 18 | 0.093 | 0.237153 | 0.341570 | -0.104417 | 18 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 126 | 0.649 | 0.164700 | 0.222457 | -0.057757 | 126 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 194 | 1.000 | 0.136726 | 0.178920 | -0.042193 | 194 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `69.5s`, rows `126`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.0988`, XGBoost `0.2869`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1114`, XGBoost `0.2881`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1127`, XGBoost `0.2888`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1168`, XGBoost `0.2881`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1221`, XGBoost `0.2871`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1296`, XGBoost `0.2827`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.1308`, XGBoost `0.2821`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.1345`, XGBoost `0.2849`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1309`, XGBoost `0.2787`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.1332`, XGBoost `0.2775`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `20.0`, recent_utility `0`
