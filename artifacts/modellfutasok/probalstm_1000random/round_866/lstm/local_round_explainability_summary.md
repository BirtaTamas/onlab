# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `100693`, seconds `23.50`, LSTM `0.5357`, delta `+0.2293`
- tick `100789`, seconds `25.00`, LSTM `0.8034`, delta `+0.1935`
- tick `103605`, seconds `69.00`, LSTM `0.8898`, delta `+0.1731`
- tick `100725`, seconds `24.00`, LSTM `0.5903`, delta `+0.0545`
- tick `100597`, seconds `22.00`, LSTM `0.3356`, delta `-0.0476`
- tick `101461`, seconds `35.50`, LSTM `0.7609`, delta `+0.0378`
- tick `103285`, seconds `64.00`, LSTM `0.7637`, delta `-0.0352`
- tick `100949`, seconds `27.50`, LSTM `0.7944`, delta `-0.0335`
- tick `101045`, seconds `29.00`, LSTM `0.7488`, delta `-0.0305`
- tick `103637`, seconds `69.50`, LSTM `0.9187`, delta `+0.0289`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003251`, |coef| `0.003251`
- `lag_00__kill_diff_last_3s`: coefficient `0.002710`, |coef| `0.002710`
- `lag_00__CT4__is_scoped`: coefficient `-0.002538`, |coef| `0.002538`
- `lag_00__damage_diff_last_5s`: coefficient `0.002516`, |coef| `0.002516`
- `lag_00__CT_damage_last_5s`: coefficient `0.002487`, |coef| `0.002487`
- `lag_14__CT_place_PIT`: coefficient `0.001975`, |coef| `0.001975`
- `lag_11__CT4__is_scoped`: coefficient `0.001676`, |coef| `0.001676`
- `lag_07__CT4__flash_duration`: coefficient `-0.001660`, |coef| `0.001660`
- `lag_04__CT4__flash_duration`: coefficient `-0.001638`, |coef| `0.001638`
- `lag_00__CT_scoped_count`: coefficient `-0.001595`, |coef| `0.001595`
- `lag_00__CT2__is_walking`: coefficient `0.001540`, |coef| `0.001540`
- `lag_06__CT_active_infernos`: coefficient `-0.001523`, |coef| `0.001523`
- `lag_01__CT_place_ARCH`: coefficient `-0.001468`, |coef| `0.001468`
- `lag_06__CT1__is_walking`: coefficient `0.001443`, |coef| `0.001443`
- `lag_13__CT4__flash_duration`: coefficient `0.001427`, |coef| `0.001427`

## Top 10 utility ridge features

- `lag_07__CT4__flash_duration`: coefficient `-0.001660` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.001638` (lowers CT win probability)
- `lag_06__CT_active_infernos`: coefficient `-0.001523` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.001427` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.001328` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.001232` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.001159` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.001108` (lowers CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.001095` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.001080` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003251` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002710` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.002538` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002516` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002487` (raises CT win probability)
- `lag_14__CT_place_PIT`: coefficient `0.001975` (raises CT win probability)
- `lag_11__CT4__is_scoped`: coefficient `0.001676` (raises CT win probability)
- `lag_00__CT_scoped_count`: coefficient `-0.001595` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `0.001540` (raises CT win probability)
- `lag_01__CT_place_ARCH`: coefficient `-0.001468` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `100693`, seconds `23.50`, LSTM delta `+0.2293`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009385`
- `lag_04__CT4__flash_duration`: contribution `+0.008839`
- `lag_00__CT4__is_scoped`: contribution `+0.008651`
- `lag_14__CT_place_PIT`: contribution `+0.008503`
- `lag_13__CT4__flash_duration`: contribution `+0.007703`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.008839`
- `lag_13__CT4__flash_duration`: contribution `+0.007703`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.004562`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.004348`
- `lag_06__CT_active_infernos`: contribution `+0.003509`

### tick `100789`, seconds `25.00`, LSTM delta `+0.1935`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009385`
- `lag_07__CT4__flash_duration`: contribution `+0.008958`
- `lag_00__kill_diff_last_3s`: contribution `+0.006523`
- `lag_11__CT4__is_scoped`: contribution `+0.005711`
- `lag_00__damage_diff_last_5s`: contribution `+0.005677`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.008958`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.003760`
- `lag_06__CT_active_infernos`: contribution `+0.003509`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.003407`
- `lag_08__T5__flash_duration`: contribution `+0.003237`

### tick `103605`, seconds `69.00`, LSTM delta `+0.1731`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009385`
- `lag_00__CT4__is_scoped`: contribution `+0.008651`
- `lag_00__kill_diff_last_3s`: contribution `+0.006523`
- `lag_01__CT_place_ARCH`: contribution `+0.005992`
- `lag_09__T4__is_scoped`: contribution `+0.005731`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.003278`

### tick `100725`, seconds `24.00`, LSTM delta `+0.0545`

Top all feature movements:
- `lag_15__CT4__is_scoped`: contribution `+0.004785`
- `lag_15__CT_place_PIT`: contribution `+0.004442`
- `lag_01__CT4__is_scoped`: contribution `+0.004081`
- `lag_00__CT2__is_walking`: contribution `-0.003633`
- `lag_01__CT4__duck_amount`: contribution `-0.003559`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.002289`
- `lag_02__T5__flash_duration`: contribution `+0.002167`
- `lag_05__CT4__flash_duration`: contribution `+0.001896`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001838`

### tick `100597`, seconds `22.00`, LSTM delta `-0.0476`

Top all feature movements:
- `lag_11__CT4__is_scoped`: contribution `-0.005711`
- `lag_10__CT_flashed_players`: contribution `-0.004250`
- `lag_13__T_place_TRAMP`: contribution `-0.003692`
- `lag_02__T_flashed_players`: contribution `-0.003588`
- `lag_01__T_place_TRAMP`: contribution `-0.003189`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `-0.002860`
- `lag_02__T5__flash_duration`: contribution `-0.002533`
- `lag_03__CT_active_infernos`: contribution `+0.001404`
- `lag_03__active_infernos_total`: contribution `+0.001361`
- `lag_14__CT_active_infernos`: contribution `+0.001262`
