# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `85102`, seconds `72.50`, LSTM `0.0993`, delta `-0.2780`
- tick `81742`, seconds `20.00`, LSTM `0.6376`, delta `+0.0743`
- tick `85070`, seconds `72.00`, LSTM `0.3772`, delta `-0.0693`
- tick `84846`, seconds `68.50`, LSTM `0.4906`, delta `-0.0669`
- tick `81198`, seconds `11.50`, LSTM `0.5745`, delta `+0.0611`
- tick `84686`, seconds `66.00`, LSTM `0.5794`, delta `-0.0531`
- tick `84654`, seconds `65.50`, LSTM `0.6325`, delta `-0.0455`
- tick `84430`, seconds `62.00`, LSTM `0.6853`, delta `+0.0325`
- tick `84942`, seconds `70.00`, LSTM `0.4641`, delta `-0.0306`
- tick `85550`, seconds `79.50`, LSTM `0.1026`, delta `-0.0304`

## Top 15 local ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.002801`, |coef| `0.002801`
- `lag_10__CT3__is_scoped`: coefficient `-0.001455`, |coef| `0.001455`
- `lag_12__CT_place_PIT`: coefficient `-0.001438`, |coef| `0.001438`
- `lag_14__CT3__flash_duration`: coefficient `-0.001391`, |coef| `0.001391`
- `lag_02__T_shots_fired_sum`: coefficient `-0.001359`, |coef| `0.001359`
- `lag_00__T1__flash_duration`: coefficient `0.001341`, |coef| `0.001341`
- `lag_08__CT3__flash`: coefficient `0.001314`, |coef| `0.001314`
- `lag_05__CT_place_PIT`: coefficient `0.001280`, |coef| `0.001280`
- `lag_01__CT3__flash_duration`: coefficient `-0.001256`, |coef| `0.001256`
- `lag_12__CT_place_BALCONY`: coefficient `0.001227`, |coef| `0.001227`
- `lag_00__T1__shots_fired`: coefficient `0.001193`, |coef| `0.001193`
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001191`, |coef| `0.001191`
- `lag_00__T2__duck_amount`: coefficient `-0.001174`, |coef| `0.001174`
- `lag_00__kill_diff_last_3s`: coefficient `0.001131`, |coef| `0.001131`
- `lag_00__T_kills_last_3s`: coefficient `-0.001128`, |coef| `0.001128`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.002801` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.001391` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001341` (raises CT win probability)
- `lag_08__CT3__flash`: coefficient `0.001314` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.001256` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001191` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.001086` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.001034` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.001000` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.000968` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT3__is_scoped`: coefficient `-0.001455` (lowers CT win probability)
- `lag_12__CT_place_PIT`: coefficient `-0.001438` (lowers CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.001359` (lowers CT win probability)
- `lag_05__CT_place_PIT`: coefficient `0.001280` (raises CT win probability)
- `lag_12__CT_place_BALCONY`: coefficient `0.001227` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `0.001193` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001174` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001131` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001128` (lowers CT win probability)
- `lag_02__T1__shots_fired`: coefficient `-0.001126` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `85102`, seconds `72.50`, LSTM delta `-0.2780`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.025376`
- `lag_14__CT3__flash_duration`: contribution `-0.008241`
- `lag_12__CT_place_BALCONY`: contribution `-0.007875`
- `lag_00__T1__shots_fired`: contribution `-0.007845`
- `lag_10__CT4__flash_duration`: contribution `-0.006936`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.025376`
- `lag_14__CT3__flash_duration`: contribution `-0.008241`
- `lag_10__CT4__flash_duration`: contribution `-0.006936`
- `lag_08__CT3__flash_duration`: contribution `-0.005378`
- `lag_08__CT3__flash`: contribution `-0.004849`

### tick `81742`, seconds `20.00`, LSTM delta `+0.0743`

Top all feature movements:
- `lag_01__CT3__flash_duration`: contribution `+0.006109`
- `lag_14__T_flashed_players`: contribution `-0.004945`
- `lag_07__CT5__flash_duration`: contribution `+0.004154`
- `lag_15__CT_place_BALCONY`: contribution `+0.004005`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.003761`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `+0.006109`
- `lag_07__CT5__flash_duration`: contribution `+0.004154`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.003761`
- `lag_15__CT3__flash_duration`: contribution `+0.003703`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.003573`

### tick `85070`, seconds `72.00`, LSTM delta `-0.0693`

Top all feature movements:
- `lag_07__CT2__is_scoped`: contribution `+0.004850`
- `lag_01__T_shots_fired_sum`: contribution `-0.003690`
- `lag_00__T1__shots_fired`: contribution `+0.003566`
- `lag_09__CT4__flash_duration`: contribution `-0.003273`
- `lag_04__T3__is_scoped`: contribution `-0.002976`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.003273`
- `lag_07__CT3__flash`: contribution `-0.002965`
- `lag_13__CT3__flash_duration`: contribution `-0.002954`
- `lag_07__CT3__flash_duration`: contribution `-0.002806`
- `lag_13__T1__flash_duration`: contribution `+0.002651`

### tick `84846`, seconds `68.50`, LSTM delta `-0.0669`

Top all feature movements:
- `lag_00__T2__duck_amount`: contribution `-0.004487`
- `lag_00__CT2__is_scoped`: contribution `-0.003934`
- `lag_00__T_kills_last_3s`: contribution `-0.003575`
- `lag_06__T1__flash_duration`: contribution `-0.003190`
- `lag_00__CT3__flash`: contribution `-0.003066`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `-0.003190`
- `lag_00__CT3__flash`: contribution `-0.003066`
- `lag_05__T_B_site_active_infernos`: contribution `-0.002923`
- `lag_08__CT3__flash_duration`: contribution `-0.002532`
- `lag_00__CT3__flash_duration`: contribution `+0.002017`

### tick `81198`, seconds `11.50`, LSTM delta `+0.0611`

Top all feature movements:
- `lag_09__CT_place_ARCH`: contribution `+0.003823`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.003382`
- `lag_08__T_place_TRAMP`: contribution `+0.003335`
- `lag_03__CT5__flash_duration`: contribution `+0.003015`
- `lag_03__CT_flashed_players`: contribution `+0.002471`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.003382`
- `lag_03__CT5__flash_duration`: contribution `+0.003015`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.002076`
- `lag_03__CT_flash_duration_sum`: contribution `+0.001734`
