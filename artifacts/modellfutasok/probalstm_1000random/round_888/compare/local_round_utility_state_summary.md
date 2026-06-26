# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `30`
- rows: `118`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 118 | 1.000 | 0.691227 | 0.704328 | -0.013101 | 40 | 78 | 1.000000 | 1.000000 |
| active/recent utility | 118 | 1.000 | 0.691227 | 0.704328 | -0.013101 | 40 | 78 | 1.000000 | 1.000000 |
| strong utility action | 111 | 0.941 | 0.684297 | 0.698683 | -0.014385 | 37 | 74 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.169 | 0.647231 | 0.662765 | -0.015535 | 10 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 92 | 0.780 | 0.689359 | 0.714869 | -0.025511 | 21 | 71 | 1.000000 | 1.000000 |
| recent utility last 5s | 30 | 0.254 | 0.664551 | 0.632206 | 0.032346 | 25 | 5 | 1.000000 | 1.000000 |
| flash effect present | 118 | 1.000 | 0.691227 | 0.704328 | -0.013101 | 40 | 78 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `45.0s`, rows `78`
- `50.0s` - `56.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.5`, LSTM `0.6095`, XGBoost `0.7155`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6093`, XGBoost `0.7150`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.7634`, XGBoost `0.8683`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.7644`, XGBoost `0.8663`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.7721`, XGBoost `0.8663`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6219`, XGBoost `0.7155`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.6226`, XGBoost `0.7155`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.7782`, XGBoost `0.8663`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6283`, XGBoost `0.7155`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5989`, XGBoost `0.6832`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
