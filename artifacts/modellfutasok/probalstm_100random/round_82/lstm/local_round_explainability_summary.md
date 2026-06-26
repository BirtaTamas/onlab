# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `70991`, seconds `11.50`, LSTM `0.5892`, delta `+0.2477`
- tick `73583`, seconds `52.00`, LSTM `0.9017`, delta `+0.1646`
- tick `70895`, seconds `10.00`, LSTM `0.4040`, delta `-0.1602`
- tick `70927`, seconds `10.50`, LSTM `0.3355`, delta `-0.0684`
- tick `71087`, seconds `13.00`, LSTM `0.6631`, delta `+0.0509`
- tick `73967`, seconds `58.00`, LSTM `0.9739`, delta `+0.0435`
- tick `73039`, seconds `43.50`, LSTM `0.7828`, delta `+0.0310`
- tick `73679`, seconds `53.50`, LSTM `0.9642`, delta `+0.0283`
- tick `72815`, seconds `40.00`, LSTM `0.7446`, delta `-0.0280`
- tick `73615`, seconds `52.50`, LSTM `0.9292`, delta `+0.0275`

## Top 15 local ridge features

- `lag_03__CT_flashes_last_5s`: coefficient `0.002254`, |coef| `0.002254`
- `lag_10__CT_he_last_5s`: coefficient `-0.001907`, |coef| `0.001907`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001607`, |coef| `0.001607`
- `lag_00__kill_diff_last_3s`: coefficient `0.001542`, |coef| `0.001542`
- `lag_00__CT_kills_last_3s`: coefficient `0.001378`, |coef| `0.001378`
- `lag_07__CT_he_last_5s`: coefficient `0.001360`, |coef| `0.001360`
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `0.001178`, |coef| `0.001178`
- `lag_01__T3__is_scoped`: coefficient `0.001131`, |coef| `0.001131`
- `lag_02__CT_flashes_last_5s`: coefficient `0.001121`, |coef| `0.001121`
- `lag_15__CT_place_UNDERA`: coefficient `-0.001103`, |coef| `0.001103`
- `lag_11__T3__is_scoped`: coefficient `0.001067`, |coef| `0.001067`
- `lag_09__CT_place_LONGA`: coefficient `-0.001059`, |coef| `0.001059`
- `lag_07__T4__flash_duration`: coefficient `0.001042`, |coef| `0.001042`
- `lag_04__T2__duck_amount`: coefficient `-0.001011`, |coef| `0.001011`
- `lag_04__CT_flashes_last_5s`: coefficient `0.001006`, |coef| `0.001006`

## Top 10 utility ridge features

- `lag_03__CT_flashes_last_5s`: coefficient `0.002254` (raises CT win probability)
- `lag_10__CT_he_last_5s`: coefficient `-0.001907` (lowers CT win probability)
- `lag_07__CT_he_last_5s`: coefficient `0.001360` (raises CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `0.001121` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001042` (raises CT win probability)
- `lag_04__CT_flashes_last_5s`: coefficient `0.001006` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001003` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.000961` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000818` (raises CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.000774` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001607` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001542` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001378` (raises CT win probability)
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `0.001178` (raises CT win probability)
- `lag_01__T3__is_scoped`: coefficient `0.001131` (raises CT win probability)
- `lag_15__CT_place_UNDERA`: coefficient `-0.001103` (lowers CT win probability)
- `lag_11__T3__is_scoped`: coefficient `0.001067` (raises CT win probability)
- `lag_09__CT_place_LONGA`: coefficient `-0.001059` (lowers CT win probability)
- `lag_04__T2__duck_amount`: coefficient `-0.001011` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000989` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `70991`, seconds `11.50`, LSTM delta `+0.2477`

Top all feature movements:
- `lag_10__CT_he_last_5s`: contribution `+0.034995`
- `lag_00__T_shots_fired_sum`: contribution `+0.014460`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.008226`
- `lag_07__T4__flash_duration`: contribution `+0.005808`
- `lag_14__T3__is_scoped`: contribution `+0.005731`

Top utility-only movements:
- `lag_10__CT_he_last_5s`: contribution `+0.034995`
- `lag_07__T4__flash_duration`: contribution `+0.005808`
- `lag_08__CT4__flash_duration`: contribution `+0.005511`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.004954`
- `lag_03__CT4__flash_duration`: contribution `+0.004667`

### tick `73583`, seconds `52.00`, LSTM delta `+0.1646`

Top all feature movements:
- `lag_03__CT_flashes_last_5s`: contribution `+0.024785`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007559`
- `lag_01__T3__is_scoped`: contribution `+0.007254`
- `lag_00__CT_kills_last_3s`: contribution `+0.003980`
- `lag_04__T2__duck_amount`: contribution `+0.003866`

Top utility-only movements:
- `lag_03__CT_flashes_last_5s`: contribution `+0.024785`

### tick `70895`, seconds `10.00`, LSTM delta `-0.1602`

Top all feature movements:
- `lag_07__CT_he_last_5s`: contribution `-0.024951`
- `lag_11__T3__is_scoped`: contribution `-0.006843`
- `lag_15__CT_place_UNDERA`: contribution `-0.006738`
- `lag_00__CT4__flash_duration`: contribution `-0.006344`
- `lag_00__T_shots_fired_sum`: contribution `-0.006025`

Top utility-only movements:
- `lag_07__CT_he_last_5s`: contribution `-0.024951`
- `lag_00__CT4__flash_duration`: contribution `-0.006344`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.004236`
- `lag_02__CT2__flash_duration`: contribution `-0.003081`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.002850`

### tick `70927`, seconds `10.50`, LSTM delta `-0.0684`

Top all feature movements:
- `lag_08__CT_he_last_5s`: contribution `-0.008987`
- `lag_14__T3__is_scoped`: contribution `-0.005731`
- `lag_05__CT_flashed_players`: contribution `-0.003721`
- `lag_12__T3__is_scoped`: contribution `-0.003490`
- `lag_01__CT4__flash_duration`: contribution `-0.003313`

Top utility-only movements:
- `lag_08__CT_he_last_5s`: contribution `-0.008987`
- `lag_01__CT4__flash_duration`: contribution `-0.003313`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.001965`
- `lag_00__CT4__flash_duration`: contribution `+0.001572`
- `lag_01__CT3__flash`: contribution `-0.001381`

### tick `71087`, seconds `13.00`, LSTM delta `+0.0509`

Top all feature movements:
- `lag_13__CT_he_last_5s`: contribution `+0.007827`
- `lag_15__CT_place_UNDERA`: contribution `+0.006738`
- `lag_00__kill_diff_last_3s`: contribution `+0.003712`
- `lag_03__T_shots_fired_sum`: contribution `-0.002911`
- `lag_10__T4__flash_duration`: contribution `+0.002228`

Top utility-only movements:
- `lag_13__CT_he_last_5s`: contribution `+0.007827`
- `lag_10__T4__flash_duration`: contribution `+0.002228`
- `lag_11__CT4__flash_duration`: contribution `+0.001962`
- `lag_05__CT5__flash_duration`: contribution `+0.001573`
- `lag_02__CT4__flash_duration`: contribution `+0.001309`
