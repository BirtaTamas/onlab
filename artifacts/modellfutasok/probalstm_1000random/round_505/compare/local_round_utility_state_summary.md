# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `7`
- rows: `257`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 257 | 1.000 | 0.270522 | 0.366475 | -0.095953 | 256 | 1 | 0.894942 | 0.498054 |
| active/recent utility | 257 | 1.000 | 0.270522 | 0.366475 | -0.095953 | 256 | 1 | 0.894942 | 0.498054 |
| strong utility action | 218 | 0.848 | 0.269039 | 0.371802 | -0.102763 | 217 | 1 | 0.876147 | 0.458716 |
| utility damage | 42 | 0.163 | 0.383618 | 0.453815 | -0.070197 | 42 | 0 | 0.547619 | 0.238095 |
| active smoke/inferno | 208 | 0.809 | 0.281857 | 0.388991 | -0.107134 | 207 | 1 | 0.870192 | 0.432692 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 257 | 1.000 | 0.270522 | 0.366475 | -0.095953 | 256 | 1 | 0.894942 | 0.498054 |

## Active Smoke/Inferno Intervals

- `8.5s` - `48.5s`, rows `81`
- `52.5s` - `115.5s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.0`, LSTM `0.1666`, XGBoost `0.3992`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.3917`, XGBoost `0.6239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.3976`, XGBoost `0.6239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.1764`, XGBoost `0.4020`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4265`, XGBoost `0.6503`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.4273`, XGBoost `0.6508`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4046`, XGBoost `0.6266`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.4043`, XGBoost `0.6239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.4061`, XGBoost `0.6239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.4347`, XGBoost `0.6514`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
