# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `17`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.106353 | 0.160313 | -0.053960 | 136 | 18 | 1.000000 | 1.000000 |
| active/recent utility | 154 | 1.000 | 0.106353 | 0.160313 | -0.053960 | 136 | 18 | 1.000000 | 1.000000 |
| strong utility action | 114 | 0.740 | 0.119667 | 0.173370 | -0.053703 | 96 | 18 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 114 | 0.740 | 0.119667 | 0.173370 | -0.053703 | 96 | 18 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 154 | 1.000 | 0.106353 | 0.160313 | -0.053960 | 136 | 18 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `63.5s`, rows `114`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.0566`, XGBoost `0.3474`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0572`, XGBoost `0.3431`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0617`, XGBoost `0.3474`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0667`, XGBoost `0.3399`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0483`, XGBoost `0.2743`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0538`, XGBoost `0.2705`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.0806`, XGBoost `0.2947`, closer `lstm`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0808`, XGBoost `0.2947`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.0836`, XGBoost `0.2964`, closer `lstm`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0915`, XGBoost `0.3032`, closer `lstm`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
