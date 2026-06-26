# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `6`
- rows: `190`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.151532 | 0.186206 | -0.034674 | 180 | 10 | 1.000000 | 0.973684 |
| active/recent utility | 190 | 1.000 | 0.151532 | 0.186206 | -0.034674 | 180 | 10 | 1.000000 | 0.973684 |
| strong utility action | 120 | 0.632 | 0.184182 | 0.236291 | -0.052109 | 120 | 0 | 1.000000 | 0.991667 |
| utility damage | 10 | 0.053 | 0.162995 | 0.271711 | -0.108716 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 120 | 0.632 | 0.184182 | 0.236291 | -0.052109 | 120 | 0 | 1.000000 | 0.991667 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 190 | 1.000 | 0.151532 | 0.186206 | -0.034674 | 180 | 10 | 1.000000 | 0.973684 |

## Active Smoke/Inferno Intervals

- `7.0s` - `66.5s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.1428`, XGBoost `0.2956`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1432`, XGBoost `0.2849`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1536`, XGBoost `0.2920`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1500`, XGBoost `0.2828`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1522`, XGBoost `0.2842`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.1672`, XGBoost `0.2956`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.1674`, XGBoost `0.2956`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.1580`, XGBoost `0.2857`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.4125`, XGBoost `0.5394`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1534`, XGBoost `0.2778`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `1.0`, recent_utility `0`
