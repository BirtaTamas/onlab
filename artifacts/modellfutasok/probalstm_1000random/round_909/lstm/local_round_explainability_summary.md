# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `28`

## Largest probability jumps

- tick `236651`, seconds `87.00`, LSTM `0.5027`, delta `+0.4157`
- tick `234571`, seconds `54.50`, LSTM `0.3328`, delta `-0.2498`
- tick `234635`, seconds `55.50`, LSTM `0.1181`, delta `-0.2483`
- tick `237099`, seconds `94.00`, LSTM `0.8858`, delta `+0.2367`
- tick `232907`, seconds `28.50`, LSTM `0.5419`, delta `-0.2133`
- tick `232427`, seconds `21.00`, LSTM `0.6914`, delta `+0.1823`
- tick `233067`, seconds `31.00`, LSTM `0.5714`, delta `+0.1676`
- tick `232267`, seconds `18.50`, LSTM `0.4640`, delta `+0.1277`
- tick `236683`, seconds `87.50`, LSTM `0.3969`, delta `-0.1058`
- tick `232171`, seconds `17.00`, LSTM `0.3039`, delta `-0.0952`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007908`, |coef| `0.007908`
- `lag_00__kill_diff_last_3s`: coefficient `0.007350`, |coef| `0.007350`
- `lag_00__T_kills_last_3s`: coefficient `-0.006475`, |coef| `0.006475`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.006396`, |coef| `0.006396`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.005587`, |coef| `0.005587`
- `lag_14__T_flash_alpha_mean`: coefficient `-0.004942`, |coef| `0.004942`
- `lag_00__T_damage_last_5s`: coefficient `-0.004261`, |coef| `0.004261`
- `lag_00__damage_diff_last_5s`: coefficient `0.004196`, |coef| `0.004196`
- `lag_03__T_duck_amount_mean`: coefficient `-0.004151`, |coef| `0.004151`
- `lag_00__T5__duck_amount`: coefficient `-0.004099`, |coef| `0.004099`
- `lag_15__T_bomb_zone_count`: coefficient `0.003749`, |coef| `0.003749`
- `lag_03__T_bomb_zone_count`: coefficient `-0.003489`, |coef| `0.003489`
- `lag_09__CT_place_ALLEY`: coefficient `-0.003472`, |coef| `0.003472`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003433`, |coef| `0.003433`
- `lag_00__CT4__alive`: coefficient `0.003427`, |coef| `0.003427`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.005587` (lowers CT win probability)
- `lag_14__T_flash_alpha_mean`: coefficient `-0.004942` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.002548` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.002323` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.002186` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.002107` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.002039` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.002002` (lowers CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001883` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001833` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007908` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.007350` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.006475` (lowers CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.006396` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.004261` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004196` (raises CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.004151` (lowers CT win probability)
- `lag_00__T5__duck_amount`: coefficient `-0.004099` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `0.003749` (raises CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `-0.003489` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `236651`, seconds `87.00`, LSTM delta `+0.4157`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.033895`
- `lag_03__T_duck_amount_mean`: contribution `+0.024141`
- `lag_15__T_bomb_zone_count`: contribution `+0.021825`
- `lag_03__T_bomb_zone_count`: contribution `+0.020313`
- `lag_00__kill_diff_last_3s`: contribution `+0.017691`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.033895`
- `lag_11__T_A_site_active_infernos`: contribution `+0.007584`

### tick `234571`, seconds `54.50`, LSTM delta `-0.2498`

Top all feature movements:
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.025745`
- `lag_00__T_kills_last_3s`: contribution `-0.020514`
- `lag_00__kill_diff_last_3s`: contribution `-0.017691`
- `lag_00__T5__duck_amount`: contribution `-0.014835`
- `lag_14__T4__is_scoped`: contribution `-0.013267`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.006181`

### tick `234635`, seconds `55.50`, LSTM delta `-0.2483`

Top all feature movements:
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.025745`
- `lag_00__T_kills_last_3s`: contribution `-0.020514`
- `lag_00__kill_diff_last_3s`: contribution `-0.017691`
- `lag_00__T5__duck_amount`: contribution `-0.015564`
- `lag_00__T_shots_fired_sum`: contribution `-0.010121`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.005034`

### tick `237099`, seconds `94.00`, LSTM delta `+0.2367`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.076657`
- `lag_14__T_flash_alpha_mean`: contribution `+0.029985`
- `lag_13__CT_duck_amount_mean`: contribution `+0.017600`
- `lag_09__CT_place_ALLEY`: contribution `+0.008789`
- `lag_14__T4__shots_fired`: contribution `+0.007882`

Top utility-only movements:
- `lag_14__T_flash_alpha_mean`: contribution `+0.029985`
- `lag_14__T4__smoke`: contribution `+0.003754`
- `lag_08__T_utility_damage_last_5s`: contribution `+0.003033`

### tick `232907`, seconds `28.50`, LSTM delta `-0.2133`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.020514`
- `lag_00__kill_diff_last_3s`: contribution `-0.017691`
- `lag_00__T_damage_last_5s`: contribution `-0.010216`
- `lag_15__CT_place_TSIDEUPPER`: contribution `-0.009587`
- `lag_00__damage_diff_last_5s`: contribution `-0.009467`

Top utility-only movements:
- `lag_12__CT_B_site_active_infernos`: contribution `-0.003735`
