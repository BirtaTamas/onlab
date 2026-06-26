# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `5`

## Largest probability jumps

- tick `35730`, seconds `67.50`, LSTM `0.2507`, delta `-0.2741`
- tick `35826`, seconds `69.00`, LSTM `0.0742`, delta `-0.0675`
- tick `35794`, seconds `68.50`, LSTM `0.1417`, delta `-0.0579`
- tick `32306`, seconds `14.00`, LSTM `0.3545`, delta `-0.0541`
- tick `35762`, seconds `68.00`, LSTM `0.1996`, delta `-0.0510`
- tick `32402`, seconds `15.50`, LSTM `0.4522`, delta `+0.0503`
- tick `32082`, seconds `10.50`, LSTM `0.4703`, delta `-0.0467`
- tick `32338`, seconds `14.50`, LSTM `0.3940`, delta `+0.0395`
- tick `32114`, seconds `11.00`, LSTM `0.4393`, delta `-0.0310`
- tick `36146`, seconds `74.00`, LSTM `0.0391`, delta `-0.0296`

## Top 15 local ridge features

- `lag_03__CT_shots_fired_sum`: coefficient `0.002348`, |coef| `0.002348`
- `lag_03__CT1__shots_fired`: coefficient `0.001978`, |coef| `0.001978`
- `lag_01__CT4__shots_fired`: coefficient `-0.001589`, |coef| `0.001589`
- `lag_02__CT_flash_duration_sum`: coefficient `-0.001453`, |coef| `0.001453`
- `lag_07__CT_shots_fired_sum`: coefficient `-0.001358`, |coef| `0.001358`
- `lag_02__T_macro_B`: coefficient `-0.001304`, |coef| `0.001304`
- `lag_02__T_place_BOMBSITEB`: coefficient `-0.001304`, |coef| `0.001304`
- `lag_00__CT4__flash_duration`: coefficient `0.001299`, |coef| `0.001299`
- `lag_02__T_place_BACKOFB`: coefficient `0.001224`, |coef| `0.001224`
- `lag_02__CT4__flash_duration`: coefficient `-0.001207`, |coef| `0.001207`
- `lag_00__CT4__shots_fired`: coefficient `-0.001184`, |coef| `0.001184`
- `lag_06__CT_shots_fired_sum`: coefficient `-0.001170`, |coef| `0.001170`
- `lag_02__CT2__flash_duration`: coefficient `-0.001159`, |coef| `0.001159`
- `lag_02__CT_flashed_players`: coefficient `-0.001156`, |coef| `0.001156`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001156`, |coef| `0.001156`

## Top 10 utility ridge features

- `lag_02__CT_flash_duration_sum`: coefficient `-0.001453` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001299` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.001207` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `-0.001159` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000899` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000810` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000758` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000730` (lowers CT win probability)
- `lag_14__T5__smoke`: coefficient `0.000594` (raises CT win probability)
- `lag_02__T2__molly`: coefficient `0.000592` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_shots_fired_sum`: coefficient `0.002348` (raises CT win probability)
- `lag_03__CT1__shots_fired`: coefficient `0.001978` (raises CT win probability)
- `lag_01__CT4__shots_fired`: coefficient `-0.001589` (lowers CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `-0.001358` (lowers CT win probability)
- `lag_02__T_macro_B`: coefficient `-0.001304` (lowers CT win probability)
- `lag_02__T_place_BOMBSITEB`: coefficient `-0.001304` (lowers CT win probability)
- `lag_02__T_place_BACKOFB`: coefficient `0.001224` (raises CT win probability)
- `lag_00__CT4__shots_fired`: coefficient `-0.001184` (lowers CT win probability)
- `lag_06__CT_shots_fired_sum`: coefficient `-0.001170` (lowers CT win probability)
- `lag_02__CT_flashed_players`: coefficient `-0.001156` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `35730`, seconds `67.50`, LSTM delta `-0.2741`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `-0.035881`
- `lag_03__CT1__shots_fired`: contribution `-0.022994`
- `lag_02__CT_flash_duration_sum`: contribution `-0.012016`
- `lag_00__CT4__flash_duration`: contribution `-0.009981`
- `lag_02__CT4__flash_duration`: contribution `-0.009279`

Top utility-only movements:
- `lag_02__CT_flash_duration_sum`: contribution `-0.012016`
- `lag_00__CT4__flash_duration`: contribution `-0.009981`
- `lag_02__CT4__flash_duration`: contribution `-0.009279`
- `lag_02__CT2__flash_duration`: contribution `-0.005997`
- `lag_02__CT1__flash_duration`: contribution `-0.004964`

### tick `35826`, seconds `69.00`, LSTM delta `-0.0675`

Top all feature movements:
- `lag_00__CT_place_DUMPSTER`: contribution `-0.045712`
- `lag_06__CT_shots_fired_sum`: contribution `+0.017889`
- `lag_03__CT_shots_fired_sum`: contribution `-0.011417`
- `lag_06__CT1__shots_fired`: contribution `+0.010621`
- `lag_07__CT_shots_fired_sum`: contribution `-0.004716`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `+0.002741`
- `lag_03__CT_flash_duration_sum`: contribution `+0.001784`

### tick `35794`, seconds `68.50`, LSTM delta `-0.0579`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `+0.011417`
- `lag_05__CT_shots_fired_sum`: contribution `+0.011377`
- `lag_02__CT4__flash_duration`: contribution `+0.009279`
- `lag_07__CT_shots_fired_sum`: contribution `-0.006603`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.006296`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.009279`
- `lag_02__CT_flash_duration_sum`: contribution `+0.004996`
- `lag_04__CT_flash_duration_sum`: contribution `-0.004695`
- `lag_04__CT4__flash_duration`: contribution `-0.003669`

### tick `32306`, seconds `14.00`, LSTM delta `-0.0541`

Top all feature movements:
- `lag_04__CT_place_ELECTRICALBOX`: contribution `-0.005028`
- `lag_15__T_place_DUMPSTER`: contribution `-0.004780`
- `lag_13__T_place_DUMPSTER`: contribution `-0.004738`
- `lag_07__CT1__flash_duration`: contribution `-0.004198`
- `lag_04__CT_place_BACKOFB`: contribution `-0.003062`

Top utility-only movements:
- `lag_07__CT1__flash_duration`: contribution `-0.004198`
- `lag_00__T4__flash_duration`: contribution `-0.001161`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.001004`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.000822`

### tick `35762`, seconds `68.00`, LSTM delta `-0.0510`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.006759`
- `lag_06__CT_shots_fired_sum`: contribution `-0.005692`
- `lag_07__CT_shots_fired_sum`: contribution `-0.004716`
- `lag_03__CT_flash_duration_sum`: contribution `-0.004291`
- `lag_04__CT_shots_fired_sum`: contribution `+0.004136`

Top utility-only movements:
- `lag_03__CT_flash_duration_sum`: contribution `-0.004291`
- `lag_03__CT4__flash_duration`: contribution `-0.002741`
- `lag_03__CT1__flash_duration`: contribution `-0.002236`
