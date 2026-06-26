# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `7`
- rows: `177`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.056584 | 0.130641 | -0.074056 | 177 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 177 | 1.000 | 0.056584 | 0.130641 | -0.074056 | 177 | 0 | 1.000000 | 1.000000 |
| strong utility action | 120 | 0.678 | 0.077056 | 0.159011 | -0.081956 | 120 | 0 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.062 | 0.119069 | 0.237357 | -0.118288 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 120 | 0.678 | 0.077056 | 0.159011 | -0.081956 | 120 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 177 | 1.000 | 0.056584 | 0.130641 | -0.074056 | 177 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `65.5s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.0`, LSTM `0.0653`, XGBoost `0.3043`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0784`, XGBoost `0.3124`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0790`, XGBoost `0.3124`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.0709`, XGBoost `0.3043`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0756`, XGBoost `0.3076`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0858`, XGBoost `0.3124`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1360`, XGBoost `0.3178`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1619`, XGBoost `0.3334`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1484`, XGBoost `0.3185`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.1837`, XGBoost `0.3444`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
