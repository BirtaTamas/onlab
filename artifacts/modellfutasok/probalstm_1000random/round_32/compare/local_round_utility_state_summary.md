# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `18`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.687923 | 0.719259 | -0.031336 | 45 | 113 | 0.740506 | 0.740506 |
| active/recent utility | 158 | 1.000 | 0.687923 | 0.719259 | -0.031336 | 45 | 113 | 0.740506 | 0.740506 |
| strong utility action | 127 | 0.804 | 0.654661 | 0.688259 | -0.033598 | 33 | 94 | 0.677165 | 0.677165 |
| utility damage | 20 | 0.127 | 0.210456 | 0.253784 | -0.043329 | 9 | 11 | 0.000000 | 0.000000 |
| active smoke/inferno | 127 | 0.804 | 0.654661 | 0.688259 | -0.033598 | 33 | 94 | 0.677165 | 0.677165 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.687923 | 0.719259 | -0.031336 | 45 | 113 | 0.740506 | 0.740506 |

## Active Smoke/Inferno Intervals

- `6.5s` - `69.5s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.0871`, XGBoost `0.2822`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `86.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.1220`, XGBoost `0.2822`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `86.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0537`, XGBoost `0.2078`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0558`, XGBoost `0.2078`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.3384`, XGBoost `0.1990`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0789`, XGBoost `0.2162`, closer `xgboost`, smoke `5`, inferno `3`, utility_damage `52.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.1417`, XGBoost `0.2761`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `86.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0812`, XGBoost `0.2083`, closer `xgboost`, smoke `5`, inferno `4`, utility_damage `43.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.1626`, XGBoost `0.2822`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `86.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0888`, XGBoost `0.2078`, closer `xgboost`, smoke `5`, inferno `4`, utility_damage `26.0`, recent_utility `0`
