# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `5`
- rows: `159`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 159 | 1.000 | 0.551365 | 0.512885 | 0.038480 | 113 | 46 | 0.886792 | 0.754717 |
| active/recent utility | 159 | 1.000 | 0.551365 | 0.512885 | 0.038480 | 113 | 46 | 0.886792 | 0.754717 |
| strong utility action | 118 | 0.742 | 0.559019 | 0.523350 | 0.035670 | 86 | 32 | 0.889831 | 0.830508 |
| utility damage | 53 | 0.333 | 0.553021 | 0.479086 | 0.073934 | 47 | 6 | 0.754717 | 0.679245 |
| active smoke/inferno | 111 | 0.698 | 0.564684 | 0.533648 | 0.031036 | 79 | 32 | 0.918919 | 0.882883 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 159 | 1.000 | 0.551365 | 0.512885 | 0.038480 | 113 | 46 | 0.886792 | 0.754717 |

## Active Smoke/Inferno Intervals

- `9.5s` - `52.5s`, rows `87`
- `61.5s` - `68.0s`, rows `14`
- `74.5s` - `79.0s`, rows `10`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.4794`, XGBoost `0.2899`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.4285`, XGBoost `0.2721`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.4390`, XGBoost `0.2857`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4904`, XGBoost `0.3383`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `35.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.4239`, XGBoost `0.2811`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6351`, XGBoost `0.4941`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.4299`, XGBoost `0.2914`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.4068`, XGBoost `0.2727`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6259`, XGBoost `0.4941`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6285`, XGBoost `0.5002`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `24.0`, recent_utility `0`
