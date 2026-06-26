# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `16`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.629420 | 0.625038 | 0.004382 | 97 | 76 | 0.947977 | 0.930636 |
| active/recent utility | 173 | 1.000 | 0.629420 | 0.625038 | 0.004382 | 97 | 76 | 0.947977 | 0.930636 |
| strong utility action | 153 | 0.884 | 0.596934 | 0.589407 | 0.007527 | 95 | 58 | 0.941176 | 0.921569 |
| utility damage | 10 | 0.058 | 0.592426 | 0.594771 | -0.002344 | 3 | 7 | 1.000000 | 1.000000 |
| active smoke/inferno | 143 | 0.827 | 0.592063 | 0.589223 | 0.002840 | 85 | 58 | 0.937063 | 0.916084 |
| recent utility last 5s | 14 | 0.081 | 0.661539 | 0.591326 | 0.070213 | 14 | 0 | 1.000000 | 1.000000 |
| flash effect present | 173 | 1.000 | 0.629420 | 0.625038 | 0.004382 | 97 | 76 | 0.947977 | 0.930636 |

## Active Smoke/Inferno Intervals

- `6.0s` - `31.0s`, rows `51`
- `32.5s` - `78.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.1782`, XGBoost `0.3414`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1969`, XGBoost `0.3550`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.2106`, XGBoost `0.3548`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.2136`, XGBoost `0.3466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.2226`, XGBoost `0.3545`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.2232`, XGBoost `0.3548`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.2190`, XGBoost `0.3479`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.2412`, XGBoost `0.3548`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.5717`, XGBoost `0.6810`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.7148`, XGBoost `0.6135`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
