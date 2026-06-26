# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `5`
- rows: `159`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.448635 | 0.207584 | 0.605794 | 0.886792 | 0.551365 |
| xgboost | 0.487115 | 0.249728 | 0.693553 | 0.754717 | 0.512885 |

## Closer Per Tick

- lstm: `113`
- xgboost: `46`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
