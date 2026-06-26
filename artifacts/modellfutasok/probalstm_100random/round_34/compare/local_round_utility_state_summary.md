# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `11`
- rows: `234`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 234 | 1.000 | 0.224459 | 0.188702 | 0.035756 | 68 | 166 | 0.662393 | 0.897436 |
| active/recent utility | 234 | 1.000 | 0.224459 | 0.188702 | 0.035756 | 68 | 166 | 0.662393 | 0.897436 |
| strong utility action | 153 | 0.654 | 0.307644 | 0.253888 | 0.053755 | 16 | 137 | 0.542484 | 0.901961 |
| utility damage | 20 | 0.085 | 0.271175 | 0.222497 | 0.048678 | 2 | 18 | 0.850000 | 1.000000 |
| active smoke/inferno | 143 | 0.611 | 0.287417 | 0.230998 | 0.056418 | 15 | 128 | 0.580420 | 0.965035 |
| recent utility last 5s | 10 | 0.043 | 0.596888 | 0.581216 | 0.015672 | 1 | 9 | 0.000000 | 0.000000 |
| flash effect present | 234 | 1.000 | 0.224459 | 0.188702 | 0.035756 | 68 | 166 | 0.662393 | 0.897436 |

## Active Smoke/Inferno Intervals

- `9.5s` - `80.5s`, rows `143`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.5154`, XGBoost `0.3249`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.3000`, XGBoost `0.1136`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.2979`, XGBoost `0.1133`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.2657`, XGBoost `0.0913`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2381`, XGBoost `0.1030`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5416`, XGBoost `0.4135`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5392`, XGBoost `0.4121`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5262`, XGBoost `0.4039`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5388`, XGBoost `0.4173`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5343`, XGBoost `0.4135`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
