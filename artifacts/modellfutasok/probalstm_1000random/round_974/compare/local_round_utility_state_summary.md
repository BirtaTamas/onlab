# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `1`
- rows: `158`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.281108 | 0.307128 | -0.026019 | 143 | 15 | 0.493671 | 0.487342 |
| active/recent utility | 123 | 0.778 | 0.215001 | 0.243433 | -0.028432 | 118 | 5 | 0.617886 | 0.617886 |
| strong utility action | 90 | 0.570 | 0.292881 | 0.329321 | -0.036441 | 85 | 5 | 0.477778 | 0.477778 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 90 | 0.570 | 0.292881 | 0.329321 | -0.036441 | 85 | 5 | 0.477778 | 0.477778 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 76 | 0.481 | 0.028485 | 0.046609 | -0.018124 | 73 | 3 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `15.5s` - `37.0s`, rows `44`
- `39.5s` - `62.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.5`, LSTM `0.0681`, XGBoost `0.2379`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0883`, XGBoost `0.2565`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0670`, XGBoost `0.2344`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0653`, XGBoost `0.2124`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5125`, XGBoost `0.5682`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.5144`, XGBoost `0.5701`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5127`, XGBoost `0.5682`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5154`, XGBoost `0.5703`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5136`, XGBoost `0.5682`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5157`, XGBoost `0.5701`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
