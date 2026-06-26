# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `10`
- rows: `166`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 166 | 1.000 | 0.709884 | 0.749791 | -0.039907 | 47 | 119 | 0.939759 | 0.939759 |
| active/recent utility | 166 | 1.000 | 0.709884 | 0.749791 | -0.039907 | 47 | 119 | 0.939759 | 0.939759 |
| strong utility action | 142 | 0.855 | 0.712121 | 0.751089 | -0.038968 | 39 | 103 | 0.929577 | 0.929577 |
| utility damage | 22 | 0.133 | 0.517856 | 0.501288 | 0.016568 | 18 | 4 | 0.636364 | 0.636364 |
| active smoke/inferno | 130 | 0.783 | 0.726197 | 0.769186 | -0.042989 | 28 | 102 | 0.923077 | 0.923077 |
| recent utility last 5s | 11 | 0.066 | 0.533794 | 0.521199 | 0.012596 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 166 | 1.000 | 0.709884 | 0.749791 | -0.039907 | 47 | 119 | 0.939759 | 0.939759 |

## Active Smoke/Inferno Intervals

- `10.0s` - `74.5s`, rows `130`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.7325`, XGBoost `0.8758`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.7417`, XGBoost `0.8748`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.7440`, XGBoost `0.8758`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.7502`, XGBoost `0.8802`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.7528`, XGBoost `0.8748`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.7592`, XGBoost `0.8759`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.7694`, XGBoost `0.8806`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.7778`, XGBoost `0.8785`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.7827`, XGBoost `0.8785`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.7857`, XGBoost `0.8806`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
