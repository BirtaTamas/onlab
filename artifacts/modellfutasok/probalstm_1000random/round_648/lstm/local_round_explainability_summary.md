# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `69621`, seconds `31.00`, LSTM `0.5505`, delta `+0.3274`
- tick `70677`, seconds `47.50`, LSTM `0.1028`, delta `-0.2700`
- tick `68373`, seconds `11.50`, LSTM `0.3252`, delta `-0.2426`
- tick `70453`, seconds `44.00`, LSTM `0.0931`, delta `-0.2400`
- tick `70517`, seconds `45.00`, LSTM `0.2821`, delta `+0.1775`
- tick `69717`, seconds `32.50`, LSTM `0.3941`, delta `-0.1323`
- tick `68469`, seconds `13.00`, LSTM `0.2380`, delta `-0.0797`
- tick `69525`, seconds `29.50`, LSTM `0.2955`, delta `-0.0649`
- tick `70581`, seconds `46.00`, LSTM `0.3445`, delta `+0.0596`
- tick `69397`, seconds `27.50`, LSTM `0.3149`, delta `+0.0594`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002608`, |coef| `0.002608`
- `lag_03__T_place_JUNGLE`: coefficient `0.002198`, |coef| `0.002198`
- `lag_00__kill_diff_last_3s`: coefficient `0.002171`, |coef| `0.002171`
- `lag_00__T4__is_scoped`: coefficient `-0.001953`, |coef| `0.001953`
- `lag_06__T_flashed_players`: coefficient `-0.001814`, |coef| `0.001814`
- `lag_00__CT_place_UNDERPASS`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_00__T_kills_last_3s`: coefficient `-0.001671`, |coef| `0.001671`
- `lag_07__CT1__flash_duration`: coefficient `0.001576`, |coef| `0.001576`
- `lag_00__damage_diff_last_5s`: coefficient `0.001563`, |coef| `0.001563`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001534`, |coef| `0.001534`
- `lag_03__CT1__flash_duration`: coefficient `-0.001515`, |coef| `0.001515`
- `lag_14__T_place_SCAFFOLDING`: coefficient `-0.001473`, |coef| `0.001473`
- `lag_12__CT_flashes_last_5s`: coefficient `0.001459`, |coef| `0.001459`
- `lag_00__T5__shots_fired`: coefficient `-0.001409`, |coef| `0.001409`
- `lag_12__CT_place_SHOP`: coefficient `-0.001401`, |coef| `0.001401`

## Top 10 utility ridge features

- `lag_07__CT1__flash_duration`: coefficient `0.001576` (raises CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.001515` (lowers CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `0.001459` (raises CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.001388` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.001317` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.001285` (raises CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.001197` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.001178` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.001165` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001104` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002608` (lowers CT win probability)
- `lag_03__T_place_JUNGLE`: coefficient `0.002198` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002171` (raises CT win probability)
- `lag_00__T4__is_scoped`: coefficient `-0.001953` (lowers CT win probability)
- `lag_06__T_flashed_players`: coefficient `-0.001814` (lowers CT win probability)
- `lag_00__CT_place_UNDERPASS`: coefficient `-0.001688` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001671` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001563` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001534` (raises CT win probability)
- `lag_14__T_place_SCAFFOLDING`: coefficient `-0.001473` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `69621`, seconds `31.00`, LSTM delta `+0.3274`

Top all feature movements:
- `lag_14__T_place_SCAFFOLDING`: contribution `+0.050160`
- `lag_15__T_place_SCAFFOLDING`: contribution `+0.039937`
- `lag_03__T_place_JUNGLE`: contribution `+0.028472`
- `lag_00__CT_place_UNDERPASS`: contribution `+0.009787`
- `lag_00__T4__is_scoped`: contribution `+0.009071`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.007165`
- `lag_12__T_A_site_active_infernos`: contribution `+0.003004`

### tick `70677`, seconds `47.50`, LSTM delta `-0.2700`

Top all feature movements:
- `lag_06__T_flashed_players`: contribution `-0.014002`
- `lag_06__T_place_JUNGLE`: contribution `-0.012512`
- `lag_07__CT1__flash_duration`: contribution `-0.012202`
- `lag_05__T_shots_fired_sum`: contribution `-0.010305`
- `lag_00__T4__is_scoped`: contribution `-0.009071`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `-0.012202`
- `lag_10__CT5__flash_duration`: contribution `-0.008559`
- `lag_10__CT1__flash_duration`: contribution `-0.008357`
- `lag_10__CT_flash_duration_sum`: contribution `-0.006943`
- `lag_06__T4__flash_duration`: contribution `-0.005221`

### tick `68373`, seconds `11.50`, LSTM delta `-0.2426`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.019550`
- `lag_02__CT_shots_fired_sum`: contribution `-0.017049`
- `lag_12__CT_flashes_last_5s`: contribution `-0.016045`
- `lag_12__CT_place_SHOP`: contribution `-0.014057`
- `lag_02__CT1__shots_fired`: contribution `-0.011675`

Top utility-only movements:
- `lag_12__CT_flashes_last_5s`: contribution `-0.016045`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.004110`
- `lag_04__T1__flash_duration`: contribution `-0.003122`

### tick `70453`, seconds `44.00`, LSTM delta `-0.2400`

Top all feature movements:
- `lag_03__CT1__flash_duration`: contribution `-0.011732`
- `lag_00__T_shots_fired_sum`: contribution `-0.009775`
- `lag_03__CT5__flash_duration`: contribution `-0.008425`
- `lag_09__CT4__flash_duration`: contribution `-0.008253`
- `lag_03__CT_flash_duration_sum`: contribution `-0.007480`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.011732`
- `lag_03__CT5__flash_duration`: contribution `-0.008425`
- `lag_09__CT4__flash_duration`: contribution `-0.008253`
- `lag_03__CT_flash_duration_sum`: contribution `-0.007480`
- `lag_00__CT1__flash_duration`: contribution `-0.006481`

### tick `70517`, seconds `45.00`, LSTM delta `+0.1775`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.023460`
- `lag_01__T_place_JUNGLE`: contribution `+0.011143`
- `lag_00__T4__is_scoped`: contribution `+0.009071`
- `lag_11__CT4__flash_duration`: contribution `+0.008918`
- `lag_01__T_flashed_players`: contribution `+0.008251`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `+0.008918`
- `lag_01__T_flash_duration_sum`: contribution `+0.005147`
- `lag_02__CT1__flash_duration`: contribution `+0.003679`
- `lag_05__CT5__flash_duration`: contribution `+0.003351`
- `lag_01__T4__flash_duration`: contribution `+0.002951`
