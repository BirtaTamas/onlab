# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `6`
- rows: `129`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 129 | 1.000 | 0.892797 | 0.978981 | -0.086184 | 0 | 129 | 1.000000 | 1.000000 |
| active/recent utility | 129 | 1.000 | 0.892797 | 0.978981 | -0.086184 | 0 | 129 | 1.000000 | 1.000000 |
| strong utility action | 100 | 0.775 | 0.878321 | 0.976576 | -0.098255 | 0 | 100 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.682 | 0.872958 | 0.977165 | -0.104208 | 0 | 88 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.155 | 0.918457 | 0.974093 | -0.055636 | 0 | 20 | 1.000000 | 1.000000 |
| flash effect present | 129 | 1.000 | 0.892797 | 0.978981 | -0.086184 | 0 | 129 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `52.0s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.8254`, XGBoost `0.9773`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.8290`, XGBoost `0.9773`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.8384`, XGBoost `0.9767`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.8387`, XGBoost `0.9766`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.8413`, XGBoost `0.9770`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.8427`, XGBoost `0.9777`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.8438`, XGBoost `0.9767`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.8444`, XGBoost `0.9770`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.8441`, XGBoost `0.9767`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.8461`, XGBoost `0.9770`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
