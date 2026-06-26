# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `19`
- rows: `226`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 226 | 1.000 | 0.354904 | 0.398103 | -0.043199 | 177 | 49 | 0.818584 | 0.818584 |
| active/recent utility | 226 | 1.000 | 0.354904 | 0.398103 | -0.043199 | 177 | 49 | 0.818584 | 0.818584 |
| strong utility action | 135 | 0.597 | 0.331553 | 0.390946 | -0.059393 | 120 | 15 | 0.844444 | 0.740741 |
| utility damage | 10 | 0.044 | 0.337501 | 0.376056 | -0.038555 | 7 | 3 | 0.800000 | 1.000000 |
| active smoke/inferno | 125 | 0.553 | 0.320699 | 0.379771 | -0.059072 | 110 | 15 | 0.832000 | 0.800000 |
| recent utility last 5s | 10 | 0.044 | 0.467231 | 0.530635 | -0.063404 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 226 | 1.000 | 0.354904 | 0.398103 | -0.043199 | 177 | 49 | 0.818584 | 0.818584 |

## Active Smoke/Inferno Intervals

- `6.5s` - `31.0s`, rows `50`
- `58.5s` - `95.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.0`, LSTM `0.2307`, XGBoost `0.4659`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.2728`, XGBoost `0.4703`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1564`, XGBoost `0.3326`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0366`, XGBoost `0.2106`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0415`, XGBoost `0.2106`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0431`, XGBoost `0.2101`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.3298`, XGBoost `0.4945`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0436`, XGBoost `0.2075`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.3387`, XGBoost `0.5024`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.3478`, XGBoost `0.5024`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
