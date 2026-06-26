# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `20`

## Largest probability jumps

- tick `154328`, seconds `52.00`, LSTM `0.8404`, delta `+0.2580`
- tick `152472`, seconds `23.00`, LSTM `0.5271`, delta `-0.1580`
- tick `152760`, seconds `27.50`, LSTM `0.6827`, delta `+0.1543`
- tick `152248`, seconds `19.50`, LSTM `0.6083`, delta `-0.1538`
- tick `152088`, seconds `17.00`, LSTM `0.7254`, delta `+0.0988`
- tick `152152`, seconds `18.00`, LSTM `0.7631`, delta `+0.0645`
- tick `152440`, seconds `22.50`, LSTM `0.6851`, delta `+0.0644`
- tick `155128`, seconds `64.50`, LSTM `0.8357`, delta `-0.0519`
- tick `152376`, seconds `21.50`, LSTM `0.6528`, delta `+0.0445`
- tick `152952`, seconds `30.50`, LSTM `0.6780`, delta `-0.0341`

## Top 15 local ridge features

- `lag_11__T_place_ARCH`: coefficient `-0.003628`, |coef| `0.003628`
- `lag_00__kill_diff_last_3s`: coefficient `0.001860`, |coef| `0.001860`
- `lag_00__CT_kills_last_3s`: coefficient `0.001824`, |coef| `0.001824`
- `lag_12__T_place_ARCH`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_00__CT_place_BANANA`: coefficient `0.001574`, |coef| `0.001574`
- `lag_05__CT3__duck_amount`: coefficient `0.001568`, |coef| `0.001568`
- `lag_00__CT3__duck_amount`: coefficient `0.001483`, |coef| `0.001483`
- `lag_00__CT_damage_last_5s`: coefficient `0.001466`, |coef| `0.001466`
- `lag_00__damage_diff_last_5s`: coefficient `0.001461`, |coef| `0.001461`
- `lag_02__CT_A_site_active_infernos`: coefficient `0.001385`, |coef| `0.001385`
- `lag_10__T_place_ARCH`: coefficient `-0.001348`, |coef| `0.001348`
- `lag_02__CT_duck_amount_mean`: coefficient `-0.001334`, |coef| `0.001334`
- `lag_14__T3__shots_fired`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001266`, |coef| `0.001266`
- `lag_07__CT_duck_amount_mean`: coefficient `0.001246`, |coef| `0.001246`

## Top 10 utility ridge features

- `lag_02__CT_A_site_active_infernos`: coefficient `0.001385` (raises CT win probability)
- `lag_03__CT3__molly`: coefficient `-0.001142` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.001063` (lowers CT win probability)
- `lag_06__CT3__smoke`: coefficient `-0.001022` (lowers CT win probability)
- `lag_02__CT_active_infernos`: coefficient `0.001007` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.000927` (lowers CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000899` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.000752` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.000676` (lowers CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.000645` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_ARCH`: coefficient `-0.003628` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001860` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001824` (raises CT win probability)
- `lag_12__T_place_ARCH`: coefficient `-0.001674` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001574` (raises CT win probability)
- `lag_05__CT3__duck_amount`: coefficient `0.001568` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001483` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001466` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001461` (raises CT win probability)
- `lag_10__T_place_ARCH`: coefficient `-0.001348` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `154328`, seconds `52.00`, LSTM delta `+0.2580`

Top all feature movements:
- `lag_11__T_place_ARCH`: contribution `+0.033749`
- `lag_02__CT_duck_amount_mean`: contribution `+0.006139`
- `lag_14__T3__shots_fired`: contribution `+0.005635`
- `lag_05__CT3__duck_amount`: contribution `+0.005629`
- `lag_00__CT3__duck_amount`: contribution `+0.005517`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `+0.004887`
- `lag_03__CT3__molly`: contribution `+0.002820`

### tick `152472`, seconds `23.00`, LSTM delta `-0.1580`

Top all feature movements:
- `lag_09__T_place_BALCONY`: contribution `-0.012173`
- `lag_06__T_place_BALCONY`: contribution `-0.010230`
- `lag_04__T2__shots_fired`: contribution `-0.007327`
- `lag_00__T1__shots_fired`: contribution `-0.006206`
- `lag_00__T_shots_fired_sum`: contribution `-0.004970`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.003894`
- `lag_08__T2__flash_duration`: contribution `-0.002556`
- `lag_08__CT5__flash_duration`: contribution `-0.002413`

### tick `152760`, seconds `27.50`, LSTM delta `+0.1543`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `+0.014929`
- `lag_13__T_shots_fired_sum`: contribution `+0.013332`
- `lag_13__T2__shots_fired`: contribution `+0.010888`
- `lag_15__T_place_BALCONY`: contribution `+0.010814`
- `lag_09__T1__shots_fired`: contribution `+0.008102`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `+0.006056`

### tick `152248`, seconds `19.50`, LSTM delta `-0.1538`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `-0.016792`
- `lag_01__T2__flash_duration`: contribution `-0.007462`
- `lag_00__CT_place_BANANA`: contribution `-0.004660`
- `lag_00__kill_diff_last_3s`: contribution `-0.004478`
- `lag_08__T5__flash_duration`: contribution `-0.003903`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `-0.007462`
- `lag_08__T5__flash_duration`: contribution `-0.003903`
- `lag_01__CT5__flash_duration`: contribution `-0.003585`
- `lag_03__T5__flash_duration`: contribution `-0.003319`
- `lag_01__T5__flash_duration`: contribution `-0.002468`

### tick `152088`, seconds `17.00`, LSTM delta `+0.0988`

Top all feature movements:
- `lag_09__CT_place_BALCONY`: contribution `+0.005623`
- `lag_00__CT_kills_last_3s`: contribution `+0.005266`
- `lag_03__T5__flash_duration`: contribution `+0.005195`
- `lag_00__kill_diff_last_3s`: contribution `+0.004478`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003518`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.005195`
