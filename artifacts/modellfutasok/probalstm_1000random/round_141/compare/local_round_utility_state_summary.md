# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `17`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.722760 | 0.752974 | -0.030214 | 57 | 173 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.722760 | 0.752974 | -0.030214 | 57 | 173 | 1.000000 | 1.000000 |
| strong utility action | 140 | 0.609 | 0.594694 | 0.623589 | -0.028895 | 56 | 84 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 130 | 0.565 | 0.600002 | 0.629710 | -0.029708 | 55 | 75 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.087 | 0.531440 | 0.602218 | -0.070778 | 1 | 19 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.722760 | 0.752974 | -0.030214 | 57 | 173 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `65.0s`, rows `116`
- `76.0s` - `82.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.0`, LSTM `0.7278`, XGBoost `0.8841`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5170`, XGBoost `0.6611`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `1`
- seconds `62.0`, LSTM `0.5307`, XGBoost `0.6687`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5211`, XGBoost `0.6586`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `1`
- seconds `62.5`, LSTM `0.5348`, XGBoost `0.6686`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.5264`, XGBoost `0.6586`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `63.5`, LSTM `0.5364`, XGBoost `0.6681`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5373`, XGBoost `0.6686`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5289`, XGBoost `0.6586`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `60.0`, LSTM `0.5292`, XGBoost `0.6586`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `1`
