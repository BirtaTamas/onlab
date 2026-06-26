# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-liquid-bo3-73g5XINyWmLhIm1c4ZyOM7/gamerlegion-vs-liquid-m1-dust2.csv`
- round_num: `12`
- rows: `194`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.666451 | 0.695984 | -0.029533 | 91 | 103 | 0.958763 | 0.881443 |
| active/recent utility | 194 | 1.000 | 0.666451 | 0.695984 | -0.029533 | 91 | 103 | 0.958763 | 0.881443 |
| strong utility action | 192 | 0.990 | 0.667534 | 0.697951 | -0.030418 | 89 | 103 | 0.958333 | 0.880208 |
| utility damage | 27 | 0.139 | 0.778622 | 0.780899 | -0.002277 | 16 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 182 | 0.938 | 0.669042 | 0.703893 | -0.034851 | 79 | 103 | 0.956044 | 0.879121 |
| recent utility last 5s | 10 | 0.052 | 0.512370 | 0.500410 | 0.011960 | 8 | 2 | 0.800000 | 0.800000 |
| flash effect present | 194 | 1.000 | 0.666451 | 0.695984 | -0.029533 | 91 | 103 | 0.958763 | 0.881443 |

## Active Smoke/Inferno Intervals

- `3.0s` - `39.5s`, rows `74`
- `43.0s` - `96.5s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.5`, LSTM `0.5665`, XGBoost `0.7741`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.5382`, XGBoost `0.7334`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5432`, XGBoost `0.7334`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5403`, XGBoost `0.7304`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.5463`, XGBoost `0.7334`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.5698`, XGBoost `0.7540`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5722`, XGBoost `0.7524`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.5738`, XGBoost `0.7540`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.5515`, XGBoost `0.7315`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5738`, XGBoost `0.7524`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
