# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `7`
- rows: `152`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 152 | 1.000 | 0.855867 | 0.817803 | 0.038064 | 119 | 33 | 1.000000 | 1.000000 |
| active/recent utility | 152 | 1.000 | 0.855867 | 0.817803 | 0.038064 | 119 | 33 | 1.000000 | 1.000000 |
| strong utility action | 129 | 0.849 | 0.867738 | 0.831238 | 0.036500 | 106 | 23 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.132 | 0.860710 | 0.849057 | 0.011654 | 18 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 129 | 0.849 | 0.867738 | 0.831238 | 0.036500 | 106 | 23 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 152 | 1.000 | 0.855867 | 0.817803 | 0.038064 | 119 | 33 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `70.5s`, rows `129`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.7229`, XGBoost `0.5478`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.7293`, XGBoost `0.5551`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.7205`, XGBoost `0.5496`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.7277`, XGBoost `0.5572`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.7047`, XGBoost `0.5504`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.6879`, XGBoost `0.5504`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.6767`, XGBoost `0.5397`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6806`, XGBoost `0.5478`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.6804`, XGBoost `0.5504`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.8978`, XGBoost `0.7780`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
