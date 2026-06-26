# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `111158`, seconds `63.00`, LSTM `0.0794`, delta `-0.1889`
- tick `112534`, seconds `84.50`, LSTM `0.0694`, delta `-0.1839`
- tick `107894`, seconds `12.00`, LSTM `0.2691`, delta `-0.1592`
- tick `112502`, seconds `84.00`, LSTM `0.2532`, delta `+0.0910`
- tick `107734`, seconds `9.50`, LSTM `0.3932`, delta `+0.0900`
- tick `112438`, seconds `83.00`, LSTM `0.1047`, delta `+0.0749`
- tick `111062`, seconds `61.50`, LSTM `0.2509`, delta `+0.0674`
- tick `107670`, seconds `8.50`, LSTM `0.3140`, delta `-0.0614`
- tick `112470`, seconds `83.50`, LSTM `0.1622`, delta `+0.0576`
- tick `107926`, seconds `12.50`, LSTM `0.2269`, delta `-0.0423`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002921`, |coef| `0.002921`
- `lag_00__CT_place_ARAMP`: coefficient `-0.002817`, |coef| `0.002817`
- `lag_00__CT_place_PIT`: coefficient `0.002565`, |coef| `0.002565`
- `lag_00__T_kills_last_3s`: coefficient `-0.002436`, |coef| `0.002436`
- `lag_00__kill_diff_last_3s`: coefficient `0.002412`, |coef| `0.002412`
- `lag_01__T4__flash_duration`: coefficient `-0.002048`, |coef| `0.002048`
- `lag_08__CT3__flash_duration`: coefficient `-0.001827`, |coef| `0.001827`
- `lag_01__T2__flash_duration`: coefficient `-0.001716`, |coef| `0.001716`
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_11__T_place_LONGA`: coefficient `-0.001526`, |coef| `0.001526`
- `lag_00__CT3__flash_duration`: coefficient `0.001499`, |coef| `0.001499`
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_10__CT3__duck_amount`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_04__T5__flash_duration`: coefficient `-0.001367`, |coef| `0.001367`
- `lag_03__CT_place_LONGA`: coefficient `-0.001351`, |coef| `0.001351`

## Top 10 utility ridge features

- `lag_01__T4__flash_duration`: coefficient `-0.002048` (lowers CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `-0.001827` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.001716` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.001537` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001499` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.001367` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.001319` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.001289` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.001207` (lowers CT win probability)
- `lag_07__CT1__molly`: coefficient `0.001108` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002921` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.002817` (lowers CT win probability)
- `lag_00__CT_place_PIT`: coefficient `0.002565` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002436` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002412` (raises CT win probability)
- `lag_11__T_place_LONGA`: coefficient `-0.001526` (lowers CT win probability)
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.001490` (lowers CT win probability)
- `lag_10__CT3__duck_amount`: coefficient `-0.001406` (lowers CT win probability)
- `lag_03__CT_place_LONGA`: coefficient `-0.001351` (lowers CT win probability)
- `lag_03__T1__duck_amount`: coefficient `0.001350` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `111158`, seconds `63.00`, LSTM delta `-0.1889`

Top all feature movements:
- `lag_01__T4__flash_duration`: contribution `-0.013639`
- `lag_00__CT_place_PIT`: contribution `-0.011045`
- `lag_08__CT3__flash_duration`: contribution `-0.008609`
- `lag_00__T_kills_last_3s`: contribution `-0.007716`
- `lag_03__CT_place_ARAMP`: contribution `-0.007092`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.013639`
- `lag_08__CT3__flash_duration`: contribution `-0.008609`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.005423`
- `lag_01__T2__flash_duration`: contribution `-0.004772`
- `lag_01__T_flash_duration_sum`: contribution `-0.004641`

### tick `112534`, seconds `84.50`, LSTM delta `-0.1839`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.030443`
- `lag_11__T_place_LONGA`: contribution `-0.013004`
- `lag_00__T_kills_last_3s`: contribution `-0.007716`
- `lag_00__kill_diff_last_3s`: contribution `-0.005805`
- `lag_06__T3__is_scoped`: contribution `-0.005273`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `-0.003895`
- `lag_12__T2__flash_duration`: contribution `-0.003322`

### tick `107894`, seconds `12.00`, LSTM delta `-0.1592`

Top all feature movements:
- `lag_04__T5__flash_duration`: contribution `-0.009550`
- `lag_03__CT_place_HOLE`: contribution `-0.008466`
- `lag_01__CT_place_HOLE`: contribution `-0.008319`
- `lag_00__T_kills_last_3s`: contribution `-0.007716`
- `lag_07__T_flashed_players`: contribution `-0.007108`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.009550`
- `lag_00__CT3__flash_duration`: contribution `-0.005233`
- `lag_07__T1__flash_duration`: contribution `-0.004918`
- `lag_07__T_flash_duration_sum`: contribution `-0.004275`
- `lag_07__T4__flash_duration`: contribution `-0.003973`

### tick `112502`, seconds `84.00`, LSTM delta `+0.0910`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012177`
- `lag_00__T2__duck_amount`: contribution `+0.004665`
- `lag_09__T_place_LONGA`: contribution `+0.004093`
- `lag_11__T2__duck_amount`: contribution `+0.003927`
- `lag_03__CT_place_LONGA`: contribution `+0.003609`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.003246`
- `lag_03__T2__flash_duration`: contribution `+0.002894`

### tick `107734`, seconds `9.50`, LSTM delta `+0.0900`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006089`
- `lag_02__CT_flashed_players`: contribution `+0.006051`
- `lag_00__kill_diff_last_3s`: contribution `+0.005805`
- `lag_00__CT3__flash_duration`: contribution `+0.005000`
- `lag_02__T1__flash_duration`: contribution `+0.003923`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.005000`
- `lag_02__T1__flash_duration`: contribution `+0.003923`
- `lag_00__CT_active_infernos`: contribution `+0.002970`
- `lag_04__CT3__molly`: contribution `+0.001730`
- `lag_00__T1__flash_duration`: contribution `+0.001551`
