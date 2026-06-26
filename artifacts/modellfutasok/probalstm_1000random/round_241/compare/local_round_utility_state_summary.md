# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `6`
- rows: `121`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 121 | 1.000 | 0.039358 | 0.051564 | -0.012206 | 99 | 22 | 1.000000 | 1.000000 |
| active/recent utility | 121 | 1.000 | 0.039358 | 0.051564 | -0.012206 | 99 | 22 | 1.000000 | 1.000000 |
| strong utility action | 91 | 0.752 | 0.045146 | 0.052772 | -0.007625 | 75 | 16 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 91 | 0.752 | 0.045146 | 0.052772 | -0.007625 | 75 | 16 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 121 | 1.000 | 0.039358 | 0.051564 | -0.012206 | 99 | 22 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `53.0s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.0382`, XGBoost `0.1235`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0468`, XGBoost `0.1244`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.1075`, XGBoost `0.0413`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0361`, XGBoost `0.0991`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0384`, XGBoost `0.0993`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1041`, XGBoost `0.0441`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0997`, XGBoost `0.0412`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0376`, XGBoost `0.0961`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0377`, XGBoost `0.0947`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0703`, XGBoost `0.1234`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
