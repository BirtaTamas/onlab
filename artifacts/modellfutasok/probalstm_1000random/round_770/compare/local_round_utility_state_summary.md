# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `14`
- rows: `209`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 209 | 1.000 | 0.288009 | 0.364916 | -0.076907 | 195 | 14 | 0.980861 | 0.971292 |
| active/recent utility | 209 | 1.000 | 0.288009 | 0.364916 | -0.076907 | 195 | 14 | 0.980861 | 0.971292 |
| strong utility action | 181 | 0.866 | 0.304028 | 0.389436 | -0.085408 | 172 | 9 | 0.977901 | 0.966851 |
| utility damage | 11 | 0.053 | 0.173804 | 0.277511 | -0.103707 | 10 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 181 | 0.866 | 0.304028 | 0.389436 | -0.085408 | 172 | 9 | 0.977901 | 0.966851 |
| recent utility last 5s | 10 | 0.048 | 0.294035 | 0.473489 | -0.179454 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 209 | 1.000 | 0.288009 | 0.364916 | -0.076907 | 195 | 14 | 0.980861 | 0.971292 |

## Active Smoke/Inferno Intervals

- `6.5s` - `51.5s`, rows `91`
- `52.5s` - `97.0s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.0`, LSTM `0.2749`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.5`, LSTM `0.2771`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `48.5`, LSTM `0.2813`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.0`, LSTM `0.2818`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `48.0`, LSTM `0.2898`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `47.0`, LSTM `0.2912`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `47.5`, LSTM `0.2931`, XGBoost `0.4737`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `42.0`, LSTM `0.3148`, XGBoost `0.4840`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.2768`, XGBoost `0.4452`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.3033`, XGBoost `0.4716`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
