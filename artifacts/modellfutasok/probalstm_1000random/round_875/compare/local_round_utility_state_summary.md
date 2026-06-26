# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `7`
- rows: `188`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.411167 | 0.410800 | 0.000367 | 103 | 85 | 0.856383 | 0.941489 |
| active/recent utility | 188 | 1.000 | 0.411167 | 0.410800 | 0.000367 | 103 | 85 | 0.856383 | 0.941489 |
| strong utility action | 151 | 0.803 | 0.432010 | 0.422135 | 0.009876 | 66 | 85 | 0.821192 | 0.927152 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 151 | 0.803 | 0.432010 | 0.422135 | 0.009876 | 66 | 85 | 0.821192 | 0.927152 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 188 | 1.000 | 0.411167 | 0.410800 | 0.000367 | 103 | 85 | 0.856383 | 0.941489 |

## Active Smoke/Inferno Intervals

- `10.0s` - `39.0s`, rows `59`
- `42.5s` - `88.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.0`, LSTM `0.4699`, XGBoost `0.2809`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4605`, XGBoost `0.2986`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.4468`, XGBoost `0.2968`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.4412`, XGBoost `0.3238`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.4864`, XGBoost `0.3774`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.4851`, XGBoost `0.3765`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.4830`, XGBoost `0.3762`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5016`, XGBoost `0.3974`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5032`, XGBoost `0.3993`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.4746`, XGBoost `0.3716`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
