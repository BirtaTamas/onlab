# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `7`
- rows: `167`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 167 | 1.000 | 0.176891 | 0.249182 | -0.072291 | 161 | 6 | 0.922156 | 0.874251 |
| active/recent utility | 167 | 1.000 | 0.176891 | 0.249182 | -0.072291 | 161 | 6 | 0.922156 | 0.874251 |
| strong utility action | 132 | 0.790 | 0.178533 | 0.222716 | -0.044183 | 126 | 6 | 0.901515 | 0.840909 |
| utility damage | 14 | 0.084 | 0.462296 | 0.490705 | -0.028410 | 9 | 5 | 0.642857 | 0.571429 |
| active smoke/inferno | 132 | 0.790 | 0.178533 | 0.222716 | -0.044183 | 126 | 6 | 0.901515 | 0.840909 |
| recent utility last 5s | 10 | 0.060 | 0.002870 | 0.030884 | -0.028015 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 167 | 1.000 | 0.176891 | 0.249182 | -0.072291 | 161 | 6 | 0.922156 | 0.874251 |

## Active Smoke/Inferno Intervals

- `7.5s` - `73.0s`, rows `132`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.2824`, XGBoost `0.4669`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2874`, XGBoost `0.4656`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3011`, XGBoost `0.4657`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.3093`, XGBoost `0.4667`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.3200`, XGBoost `0.4697`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `15.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.3175`, XGBoost `0.4667`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.3302`, XGBoost `0.4672`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.3310`, XGBoost `0.4672`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.3328`, XGBoost `0.4683`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `15.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.3343`, XGBoost `0.4667`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
