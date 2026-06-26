# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `14`
- rows: `215`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.130312 | 0.132059 | -0.001748 | 152 | 63 | 1.000000 | 1.000000 |
| active/recent utility | 215 | 1.000 | 0.130312 | 0.132059 | -0.001748 | 152 | 63 | 1.000000 | 1.000000 |
| strong utility action | 163 | 0.758 | 0.162129 | 0.157207 | 0.004922 | 100 | 63 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.047 | 0.382219 | 0.379267 | 0.002952 | 5 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 153 | 0.712 | 0.157356 | 0.142531 | 0.014825 | 90 | 63 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.047 | 0.235155 | 0.381753 | -0.146598 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 215 | 1.000 | 0.130312 | 0.132059 | -0.001748 | 152 | 63 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `61.5s`, rows `109`
- `67.0s` - `88.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.5`, LSTM `0.2145`, XGBoost `0.3774`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.5`, LSTM `0.2246`, XGBoost `0.3858`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.0`, LSTM `0.2258`, XGBoost `0.3844`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.5`, LSTM `0.2347`, XGBoost `0.3812`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `4.0`, LSTM `0.2304`, XGBoost `0.3767`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `5.0`, LSTM `0.2320`, XGBoost `0.3780`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.0`, LSTM `0.2373`, XGBoost `0.3812`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `1.5`, LSTM `0.2424`, XGBoost `0.3844`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `1.0`, LSTM `0.2487`, XGBoost `0.3842`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `0.5`, LSTM `0.2611`, XGBoost `0.3842`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
