# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `11`
- rows: `214`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.452198 | 0.479883 | -0.027685 | 97 | 117 | 0.621495 | 0.415888 |
| active/recent utility | 214 | 1.000 | 0.452198 | 0.479883 | -0.027685 | 97 | 117 | 0.621495 | 0.415888 |
| strong utility action | 187 | 0.874 | 0.449252 | 0.478784 | -0.029531 | 82 | 105 | 0.631016 | 0.395722 |
| utility damage | 40 | 0.187 | 0.481627 | 0.561490 | -0.079863 | 5 | 35 | 0.600000 | 0.650000 |
| active smoke/inferno | 177 | 0.827 | 0.445194 | 0.476443 | -0.031249 | 76 | 101 | 0.610169 | 0.361582 |
| recent utility last 5s | 10 | 0.047 | 0.521087 | 0.520220 | 0.000867 | 6 | 4 | 1.000000 | 1.000000 |
| flash effect present | 214 | 1.000 | 0.452198 | 0.479883 | -0.027685 | 97 | 117 | 0.621495 | 0.415888 |

## Active Smoke/Inferno Intervals

- `7.5s` - `73.5s`, rows `133`
- `80.0s` - `101.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `93.5`, LSTM `0.5192`, XGBoost `0.7540`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.5248`, XGBoost `0.7540`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.5423`, XGBoost `0.7540`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1064`, XGBoost `0.2867`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.1113`, XGBoost `0.2849`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.1298`, XGBoost `0.2978`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.1362`, XGBoost `0.2978`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.1364`, XGBoost `0.2973`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.1251`, XGBoost `0.2838`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.1381`, XGBoost `0.2961`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
