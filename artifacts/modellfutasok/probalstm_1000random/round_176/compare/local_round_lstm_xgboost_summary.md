# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-metizport-inferno-qyaWW06KtkktSDfICHvaab/wildcard-vs-metizport-inferno.csv`
- round_num: `7`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.264269 | 0.098396 | 0.331601 | 1.000000 | 0.735731 |
| xgboost | 0.301240 | 0.132675 | 0.397995 | 1.000000 | 0.698760 |

## Closer Per Tick

- lstm: `156`
- xgboost: `74`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
