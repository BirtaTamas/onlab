# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-spirit-bo3-NmwBJVzYbgyZgcQrbNESHr/flyquest-vs-spirit-m1-anubis.csv`
- round_num: `10`
- rows: `309`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 309 | 1.000 | 0.397172 | 0.391788 | 0.005384 | 110 | 199 | 0.825243 | 0.805825 |
| active/recent utility | 309 | 1.000 | 0.397172 | 0.391788 | 0.005384 | 110 | 199 | 0.825243 | 0.805825 |
| strong utility action | 193 | 0.625 | 0.402755 | 0.400893 | 0.001861 | 67 | 126 | 0.818653 | 0.761658 |
| utility damage | 10 | 0.032 | 0.903442 | 0.916627 | -0.013186 | 9 | 1 | 0.000000 | 0.000000 |
| active smoke/inferno | 177 | 0.573 | 0.387252 | 0.391164 | -0.003912 | 66 | 111 | 0.892655 | 0.807910 |
| recent utility last 5s | 16 | 0.052 | 0.574255 | 0.508525 | 0.065730 | 1 | 15 | 0.000000 | 0.250000 |
| flash effect present | 309 | 1.000 | 0.397172 | 0.391788 | 0.005384 | 110 | 199 | 0.825243 | 0.805825 |

## Active Smoke/Inferno Intervals

- `12.5s` - `67.5s`, rows `111`
- `95.5s` - `122.5s`, rows `55`
- `142.5s` - `147.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `106.0`, LSTM `0.2904`, XGBoost `0.6131`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.3153`, XGBoost `0.6057`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.3853`, XGBoost `0.6179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.3866`, XGBoost `0.6057`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.4100`, XGBoost `0.6224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.3776`, XGBoost `0.5860`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.3252`, XGBoost `0.5273`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.4040`, XGBoost `0.6057`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.5`, LSTM `0.3328`, XGBoost `0.5296`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.3185`, XGBoost `0.5135`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
