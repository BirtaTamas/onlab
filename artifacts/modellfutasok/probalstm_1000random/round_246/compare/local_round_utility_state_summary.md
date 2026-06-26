# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `9`
- rows: `125`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.595238 | 0.682854 | -0.087616 | 0 | 125 | 1.000000 | 1.000000 |
| active/recent utility | 125 | 1.000 | 0.595238 | 0.682854 | -0.087616 | 0 | 125 | 1.000000 | 1.000000 |
| strong utility action | 109 | 0.872 | 0.604616 | 0.701520 | -0.096904 | 0 | 109 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 109 | 0.872 | 0.604616 | 0.701520 | -0.096904 | 0 | 109 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 125 | 1.000 | 0.595238 | 0.682854 | -0.087616 | 0 | 125 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `62.0s`, rows `109`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.5549`, XGBoost `0.7603`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5549`, XGBoost `0.7587`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5560`, XGBoost `0.7595`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5570`, XGBoost `0.7602`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5569`, XGBoost `0.7591`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5608`, XGBoost `0.7602`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5540`, XGBoost `0.7529`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5600`, XGBoost `0.7587`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5543`, XGBoost `0.7527`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5557`, XGBoost `0.7539`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
