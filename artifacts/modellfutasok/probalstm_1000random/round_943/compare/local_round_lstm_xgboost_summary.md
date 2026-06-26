# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `29`
- rows: `240`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.287496 | 0.109208 | 0.364604 | 0.991667 | 0.712504 |
| xgboost | 0.283745 | 0.122429 | 0.375435 | 0.975000 | 0.716255 |

## Closer Per Tick

- lstm: `113`
- xgboost: `127`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
