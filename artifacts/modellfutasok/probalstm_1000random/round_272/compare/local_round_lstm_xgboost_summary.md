# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `9`
- rows: `203`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.321673 | 0.112844 | 0.398246 | 0.995074 | 0.678327 |
| xgboost | 0.343757 | 0.128092 | 0.433722 | 0.995074 | 0.656243 |

## Closer Per Tick

- lstm: `152`
- xgboost: `51`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
