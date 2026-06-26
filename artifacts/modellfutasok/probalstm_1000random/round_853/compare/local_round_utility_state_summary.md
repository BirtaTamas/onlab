# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `10`
- rows: `225`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 225 | 1.000 | 0.633692 | 0.657917 | -0.024226 | 70 | 155 | 0.782222 | 0.955556 |
| active/recent utility | 225 | 1.000 | 0.633692 | 0.657917 | -0.024226 | 70 | 155 | 0.782222 | 0.955556 |
| strong utility action | 163 | 0.724 | 0.643664 | 0.656039 | -0.012375 | 60 | 103 | 0.889571 | 0.950920 |
| utility damage | 10 | 0.044 | 0.723847 | 0.722778 | 0.001069 | 8 | 2 | 0.800000 | 1.000000 |
| active smoke/inferno | 163 | 0.724 | 0.643664 | 0.656039 | -0.012375 | 60 | 103 | 0.889571 | 0.950920 |
| recent utility last 5s | 10 | 0.044 | 0.678641 | 0.678677 | -0.000036 | 6 | 4 | 1.000000 | 1.000000 |
| flash effect present | 225 | 1.000 | 0.633692 | 0.657917 | -0.024226 | 70 | 155 | 0.782222 | 0.955556 |

## Active Smoke/Inferno Intervals

- `6.0s` - `87.0s`, rows `163`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.5`, LSTM `0.3539`, XGBoost `0.5109`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.3535`, XGBoost `0.4995`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.5704`, XGBoost `0.6957`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.5721`, XGBoost `0.6968`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.5726`, XGBoost `0.6962`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.3982`, XGBoost `0.5125`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.5420`, XGBoost `0.6402`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.5384`, XGBoost `0.6338`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.3976`, XGBoost `0.4858`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.5606`, XGBoost `0.6402`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
