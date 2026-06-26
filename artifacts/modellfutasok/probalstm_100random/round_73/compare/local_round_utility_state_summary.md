# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `21`
- rows: `176`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.531147 | 0.584899 | -0.053752 | 153 | 23 | 0.215909 | 0.159091 |
| active/recent utility | 176 | 1.000 | 0.531147 | 0.584899 | -0.053752 | 153 | 23 | 0.215909 | 0.159091 |
| strong utility action | 174 | 0.989 | 0.531277 | 0.584844 | -0.053567 | 152 | 22 | 0.212644 | 0.155172 |
| utility damage | 11 | 0.062 | 0.621497 | 0.675414 | -0.053916 | 11 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 162 | 0.920 | 0.517106 | 0.573841 | -0.056735 | 144 | 18 | 0.228395 | 0.166667 |
| recent utility last 5s | 26 | 0.148 | 0.702876 | 0.729519 | -0.026642 | 21 | 5 | 0.000000 | 0.000000 |
| flash effect present | 176 | 1.000 | 0.531147 | 0.584899 | -0.053752 | 153 | 23 | 0.215909 | 0.159091 |

## Active Smoke/Inferno Intervals

- `6.5s` - `76.0s`, rows `140`
- `77.0s` - `87.5s`, rows `22`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.0`, LSTM `0.2330`, XGBoost `0.4482`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.2542`, XGBoost `0.4512`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.2575`, XGBoost `0.4416`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.3564`, XGBoost `0.5344`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.3588`, XGBoost `0.5306`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.2854`, XGBoost `0.4423`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.3749`, XGBoost `0.5257`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.2774`, XGBoost `0.4253`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2824`, XGBoost `0.4253`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.3092`, XGBoost `0.4384`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
