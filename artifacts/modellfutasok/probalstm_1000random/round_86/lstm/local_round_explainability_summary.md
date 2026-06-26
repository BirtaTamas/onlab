# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `46608`, seconds `13.00`, LSTM `0.8761`, delta `+0.1230`
- tick `46352`, seconds `9.00`, LSTM `0.8153`, delta `+0.0987`
- tick `47184`, seconds `22.00`, LSTM `0.9536`, delta `+0.0512`
- tick `46192`, seconds `6.50`, LSTM `0.6896`, delta `+0.0499`
- tick `46544`, seconds `12.00`, LSTM `0.7520`, delta `-0.0383`
- tick `47760`, seconds `31.00`, LSTM `0.9769`, delta `+0.0348`
- tick `46256`, seconds `7.50`, LSTM `0.7121`, delta `+0.0183`
- tick `47376`, seconds `25.00`, LSTM `0.9372`, delta `-0.0175`
- tick `46640`, seconds `13.50`, LSTM `0.8933`, delta `+0.0172`
- tick `46448`, seconds `10.50`, LSTM `0.7925`, delta `-0.0166`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001153`, |coef| `0.001153`
- `lag_02__T1__is_scoped`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_04__T1__is_scoped`: coefficient `0.000964`, |coef| `0.000964`
- `lag_00__kill_diff_last_3s`: coefficient `0.000928`, |coef| `0.000928`
- `lag_11__T_place_PALACEALLEY`: coefficient `0.000894`, |coef| `0.000894`
- `lag_00__CT3__shots_fired`: coefficient `0.000874`, |coef| `0.000874`
- `lag_07__CT_place_SHOP`: coefficient `0.000851`, |coef| `0.000851`
- `lag_01__CT_place_UNDERPASS`: coefficient `0.000805`, |coef| `0.000805`
- `lag_05__T_place_TRAMP`: coefficient `0.000717`, |coef| `0.000717`
- `lag_09__CT_place_UNDERPASS`: coefficient `0.000697`, |coef| `0.000697`
- `lag_10__CT_place_SHOP`: coefficient `-0.000687`, |coef| `0.000687`
- `lag_12__CT_flashed_players`: coefficient `0.000663`, |coef| `0.000663`
- `lag_00__CT_damage_last_5s`: coefficient `0.000648`, |coef| `0.000648`
- `lag_01__CT3__shots_fired`: coefficient `0.000622`, |coef| `0.000622`
- `lag_05__CT_place_SNIPERSNEST`: coefficient `0.000618`, |coef| `0.000618`

## Top 10 utility ridge features

- `lag_12__T1__flash_duration`: coefficient `0.000612` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000521` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.000513` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `0.000492` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.000473` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.000440` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.000419` (raises CT win probability)
- `lag_14__CT_active_infernos`: coefficient `0.000415` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000411` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000408` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001153` (raises CT win probability)
- `lag_02__T1__is_scoped`: coefficient `-0.000979` (lowers CT win probability)
- `lag_04__T1__is_scoped`: coefficient `0.000964` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000928` (raises CT win probability)
- `lag_11__T_place_PALACEALLEY`: coefficient `0.000894` (raises CT win probability)
- `lag_00__CT3__shots_fired`: coefficient `0.000874` (raises CT win probability)
- `lag_07__CT_place_SHOP`: coefficient `0.000851` (raises CT win probability)
- `lag_01__CT_place_UNDERPASS`: coefficient `0.000805` (raises CT win probability)
- `lag_05__T_place_TRAMP`: coefficient `0.000717` (raises CT win probability)
- `lag_09__CT_place_UNDERPASS`: coefficient `0.000697` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `46608`, seconds `13.00`, LSTM delta `+0.1230`

Top all feature movements:
- `lag_02__T1__is_scoped`: contribution `+0.005596`
- `lag_04__T1__is_scoped`: contribution `+0.005507`
- `lag_09__CT_place_UNDERPASS`: contribution `+0.004041`
- `lag_10__CT_place_SHOP`: contribution `+0.003443`
- `lag_00__CT_kills_last_3s`: contribution `+0.003330`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `+0.003164`
- `lag_03__T1__flash_duration`: contribution `+0.002444`
- `lag_12__CT3__flash_duration`: contribution `+0.001855`
- `lag_05__CT3__flash_duration`: contribution `+0.001715`

### tick `46352`, seconds `9.00`, LSTM delta `+0.0987`

Top all feature movements:
- `lag_11__T_place_PALACEALLEY`: contribution `+0.006222`
- `lag_04__T1__is_scoped`: contribution `+0.005507`
- `lag_01__CT_place_UNDERPASS`: contribution `+0.004666`
- `lag_07__CT_place_SHOP`: contribution `+0.004271`
- `lag_00__CT_kills_last_3s`: contribution `+0.003330`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.002654`
- `lag_04__CT3__flash_duration`: contribution `+0.002197`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.001440`

### tick `47184`, seconds `22.00`, LSTM delta `+0.0512`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.003330`
- `lag_01__CT_place_TRUCK`: contribution `+0.003244`
- `lag_00__CT3__shots_fired`: contribution `+0.002697`
- `lag_00__kill_diff_last_3s`: contribution `+0.002234`
- `lag_08__CT5__is_scoped`: contribution `+0.001732`

Top utility-only movements:
- `lag_14__CT_active_infernos`: contribution `+0.000957`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.000736`

### tick `46192`, seconds `6.50`, LSTM delta `+0.0499`

Top all feature movements:
- `lag_06__T_place_PALACEALLEY`: contribution `+0.003576`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.002929`
- `lag_02__CT_place_SHOP`: contribution `-0.001656`
- `lag_13__CT_place_CTSPAWN`: contribution `+0.001333`
- `lag_01__CT_A_site_active_infernos`: contribution `+0.001205`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `+0.001205`
- `lag_13__CT_molly_inv`: contribution `+0.000863`
- `lag_13__CT_utility_inv`: contribution `+0.000737`
- `lag_13__CT_smoke_inv`: contribution `+0.000687`
- `lag_13__CT5__molly`: contribution `+0.000569`

### tick `46544`, seconds `12.00`, LSTM delta `-0.0383`

Top all feature movements:
- `lag_02__T1__is_scoped`: contribution `-0.005596`
- `lag_00__CT_kills_last_3s`: contribution `-0.003330`
- `lag_11__T_place_PALACEALLEY`: contribution `+0.003111`
- `lag_05__T1__is_scoped`: contribution `-0.003046`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.002929`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `-0.001693`
- `lag_10__T1__flash_duration`: contribution `-0.001304`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.001205`
- `lag_03__CT3__flash_duration`: contribution `-0.001070`
