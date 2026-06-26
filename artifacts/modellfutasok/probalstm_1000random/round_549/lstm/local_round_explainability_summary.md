# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `34`

## Largest probability jumps

- tick `299501`, seconds `97.00`, LSTM `0.2315`, delta `-0.2933`
- tick `299533`, seconds `97.50`, LSTM `0.1155`, delta `-0.1160`
- tick `298957`, seconds `88.50`, LSTM `0.4349`, delta `-0.1017`
- tick `301357`, seconds `126.00`, LSTM `0.2583`, delta `+0.0920`
- tick `301741`, seconds `132.00`, LSTM `0.0123`, delta `-0.0841`
- tick `301517`, seconds `128.50`, LSTM `0.1477`, delta `-0.0686`
- tick `299245`, seconds `93.00`, LSTM `0.5098`, delta `+0.0679`
- tick `301389`, seconds `126.50`, LSTM `0.1921`, delta `-0.0662`
- tick `300845`, seconds `118.00`, LSTM `0.2048`, delta `+0.0516`
- tick `299213`, seconds `92.50`, LSTM `0.4419`, delta `-0.0376`

## Top 15 local ridge features

- `lag_03__T_place_BALCONY`: coefficient `0.003335`, |coef| `0.003335`
- `lag_00__CT_place_QUAD`: coefficient `0.002911`, |coef| `0.002911`
- `lag_09__T_place_BALCONY`: coefficient `-0.002645`, |coef| `0.002645`
- `lag_07__T_place_BALCONY`: coefficient `-0.002326`, |coef| `0.002326`
- `lag_00__CT2__flash_duration`: coefficient `0.002172`, |coef| `0.002172`
- `lag_02__T_place_BALCONY`: coefficient `0.001862`, |coef| `0.001862`
- `lag_00__T_kills_last_3s`: coefficient `-0.001784`, |coef| `0.001784`
- `lag_08__T_place_PIT`: coefficient `-0.001729`, |coef| `0.001729`
- `lag_12__T_place_PIT`: coefficient `-0.001546`, |coef| `0.001546`
- `lag_00__kill_diff_last_3s`: coefficient `0.001494`, |coef| `0.001494`
- `lag_00__CT1__flash_duration`: coefficient `0.001435`, |coef| `0.001435`
- `lag_06__T_place_PIT`: coefficient `-0.001399`, |coef| `0.001399`
- `lag_00__T1__flash_duration`: coefficient `-0.001368`, |coef| `0.001368`
- `lag_00__damage_diff_last_5s`: coefficient `0.001358`, |coef| `0.001358`
- `lag_00__T_place_PIT`: coefficient `-0.001330`, |coef| `0.001330`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.002172` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001435` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001368` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001290` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001280` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `-0.001159` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.001065` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `-0.001064` (lowers CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.001042` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.000973` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_BALCONY`: coefficient `0.003335` (raises CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.002911` (raises CT win probability)
- `lag_09__T_place_BALCONY`: coefficient `-0.002645` (lowers CT win probability)
- `lag_07__T_place_BALCONY`: coefficient `-0.002326` (lowers CT win probability)
- `lag_02__T_place_BALCONY`: coefficient `0.001862` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001784` (lowers CT win probability)
- `lag_08__T_place_PIT`: coefficient `-0.001729` (lowers CT win probability)
- `lag_12__T_place_PIT`: coefficient `-0.001546` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001494` (raises CT win probability)
- `lag_06__T_place_PIT`: coefficient `-0.001399` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `299501`, seconds `97.00`, LSTM delta `-0.2933`

Top all feature movements:
- `lag_03__T_place_BALCONY`: contribution `-0.045868`
- `lag_09__T_place_BALCONY`: contribution `-0.036378`
- `lag_07__T_place_BALCONY`: contribution `-0.031981`
- `lag_02__T_place_BALCONY`: contribution `-0.025606`
- `lag_06__T_place_PIT`: contribution `-0.008825`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `-0.005374`
- `lag_10__CT4__flash_duration`: contribution `-0.004441`

### tick `299533`, seconds `97.50`, LSTM delta `-0.1160`

Top all feature movements:
- `lag_03__T_place_BALCONY`: contribution `-0.045868`
- `lag_10__T_place_BALCONY`: contribution `-0.015305`
- `lag_04__T_place_BALCONY`: contribution `-0.006577`
- `lag_09__T1__flash_duration`: contribution `-0.005018`
- `lag_08__T_place_BALCONY`: contribution `-0.004376`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.005018`

### tick `298957`, seconds `88.50`, LSTM delta `-0.1017`

Top all feature movements:
- `lag_00__T1__flash_duration`: contribution `-0.007051`
- `lag_00__T_kills_last_3s`: contribution `-0.005652`
- `lag_13__T5__is_scoped`: contribution `-0.004397`
- `lag_00__kill_diff_last_3s`: contribution `-0.003597`
- `lag_00__CT4__flash_duration`: contribution `-0.003160`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `-0.007051`
- `lag_00__CT4__flash_duration`: contribution `-0.003160`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.002412`
- `lag_00__CT_flash_duration_sum`: contribution `+0.002387`

### tick `301357`, seconds `126.00`, LSTM delta `+0.0920`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `+0.022942`
- `lag_00__CT2__flash_duration`: contribution `+0.015353`
- `lag_02__CT1__flash_duration`: contribution `+0.010391`
- `lag_02__CT_place_QUAD`: contribution `+0.007550`
- `lag_00__CT_flash_duration_sum`: contribution `+0.004109`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.015353`
- `lag_02__CT1__flash_duration`: contribution `+0.010391`
- `lag_00__CT_flash_duration_sum`: contribution `+0.004109`
- `lag_02__CT_flash_duration_sum`: contribution `+0.002988`
- `lag_12__CT2__flash_duration`: contribution `+0.001390`

### tick `301741`, seconds `132.00`, LSTM delta `-0.0841`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.011304`
- `lag_12__CT_place_QUAD`: contribution `-0.008873`
- `lag_00__damage_diff_last_5s`: contribution `-0.007414`
- `lag_00__kill_diff_last_3s`: contribution `-0.007194`
- `lag_14__CT1__flash_duration`: contribution `-0.006910`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.006910`
- `lag_12__CT2__flash_duration`: contribution `-0.003901`
- `lag_14__CT_flash_duration_sum`: contribution `-0.002421`
