# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `12`
- rows: `288`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 288 | 1.000 | 0.623641 | 0.618749 | 0.004893 | 110 | 178 | 0.350694 | 0.312500 |
| active/recent utility | 288 | 1.000 | 0.623641 | 0.618749 | 0.004893 | 110 | 178 | 0.350694 | 0.312500 |
| strong utility action | 178 | 0.618 | 0.596592 | 0.578283 | 0.018309 | 73 | 105 | 0.387640 | 0.325843 |
| utility damage | 20 | 0.069 | 0.797314 | 0.744353 | 0.052961 | 4 | 16 | 0.150000 | 0.000000 |
| active smoke/inferno | 178 | 0.618 | 0.596592 | 0.578283 | 0.018309 | 73 | 105 | 0.387640 | 0.325843 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 288 | 1.000 | 0.623641 | 0.618749 | 0.004893 | 110 | 178 | 0.350694 | 0.312500 |

## Active Smoke/Inferno Intervals

- `6.5s` - `67.5s`, rows `123`
- `107.0s` - `113.5s`, rows `14`
- `123.5s` - `143.5s`, rows `41`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `113.5`, LSTM `0.2307`, XGBoost `0.6745`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `113.0`, LSTM `0.2496`, XGBoost `0.6578`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.2038`, XGBoost `0.6077`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.2094`, XGBoost `0.6098`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.2236`, XGBoost `0.6077`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `112.5`, LSTM `0.2437`, XGBoost `0.6181`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.2460`, XGBoost `0.6150`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.2856`, XGBoost `0.6068`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.3416`, XGBoost `0.6032`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.3739`, XGBoost `0.5727`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `2.0`, recent_utility `0`
