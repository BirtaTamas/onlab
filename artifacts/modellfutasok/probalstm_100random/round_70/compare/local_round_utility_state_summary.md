# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv`
- round_num: `6`
- rows: `195`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 195 | 1.000 | 0.244477 | 0.253583 | -0.009106 | 136 | 59 | 0.656410 | 0.712821 |
| active/recent utility | 195 | 1.000 | 0.244477 | 0.253583 | -0.009106 | 136 | 59 | 0.656410 | 0.712821 |
| strong utility action | 134 | 0.687 | 0.300404 | 0.305773 | -0.005369 | 75 | 59 | 0.597015 | 0.686567 |
| utility damage | 24 | 0.123 | 0.263188 | 0.261467 | 0.001721 | 13 | 11 | 0.583333 | 0.583333 |
| active smoke/inferno | 134 | 0.687 | 0.300404 | 0.305773 | -0.005369 | 75 | 59 | 0.597015 | 0.686567 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 195 | 1.000 | 0.244477 | 0.253583 | -0.009106 | 136 | 59 | 0.656410 | 0.712821 |

## Active Smoke/Inferno Intervals

- `7.0s` - `73.5s`, rows `134`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.0`, LSTM `0.6325`, XGBoost `0.4981`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6221`, XGBoost `0.4981`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6216`, XGBoost `0.4981`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.0084`, XGBoost `0.1302`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6209`, XGBoost `0.5011`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0114`, XGBoost `0.1302`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.0075`, XGBoost `0.1214`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6099`, XGBoost `0.5007`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0166`, XGBoost `0.1232`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.0086`, XGBoost `0.1140`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
