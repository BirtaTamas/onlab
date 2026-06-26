# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `17`
- rows: `191`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.537016 | 0.665120 | -0.128104 | 47 | 144 | 0.675393 | 0.858639 |
| active/recent utility | 191 | 1.000 | 0.537016 | 0.665120 | -0.128104 | 47 | 144 | 0.675393 | 0.858639 |
| strong utility action | 164 | 0.859 | 0.518610 | 0.653179 | -0.134570 | 40 | 124 | 0.646341 | 0.859756 |
| utility damage | 10 | 0.052 | 0.482364 | 0.681728 | -0.199364 | 0 | 10 | 0.600000 | 1.000000 |
| active smoke/inferno | 156 | 0.817 | 0.507510 | 0.653115 | -0.145604 | 32 | 124 | 0.628205 | 0.852564 |
| recent utility last 5s | 10 | 0.052 | 0.729639 | 0.653069 | 0.076570 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 191 | 1.000 | 0.537016 | 0.665120 | -0.128104 | 47 | 144 | 0.675393 | 0.858639 |

## Active Smoke/Inferno Intervals

- `7.5s` - `31.0s`, rows `48`
- `33.5s` - `87.0s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.1498`, XGBoost `0.6419`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.1681`, XGBoost `0.6419`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1717`, XGBoost `0.6428`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.2194`, XGBoost `0.6764`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.1831`, XGBoost `0.6383`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.2249`, XGBoost `0.6764`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.2218`, XGBoost `0.6731`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.2303`, XGBoost `0.6764`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.1965`, XGBoost `0.6342`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.2152`, XGBoost `0.6305`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
