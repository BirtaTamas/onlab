# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `57053`, seconds `103.50`, LSTM `0.4023`, delta `-0.4077`
- tick `57917`, seconds `117.00`, LSTM `0.7744`, delta `+0.2398`
- tick `57725`, seconds `114.00`, LSTM `0.4917`, delta `+0.1980`
- tick `57533`, seconds `111.00`, LSTM `0.2833`, delta `+0.1757`
- tick `56925`, seconds `101.50`, LSTM `0.7385`, delta `+0.1523`
- tick `57341`, seconds `108.00`, LSTM `0.0633`, delta `-0.1515`
- tick `57085`, seconds `104.00`, LSTM `0.3111`, delta `-0.0912`
- tick `56189`, seconds `90.00`, LSTM `0.6381`, delta `-0.0686`
- tick `50941`, seconds `8.00`, LSTM `0.5896`, delta `-0.0644`
- tick `57021`, seconds `103.00`, LSTM `0.8100`, delta `+0.0587`

## Top 15 local ridge features

- `lag_14__T_place_SCAFFOLDING`: coefficient `-0.003182`, |coef| `0.003182`
- `lag_08__T_place_SCAFFOLDING`: coefficient `-0.002920`, |coef| `0.002920`
- `lag_00__T_kills_last_3s`: coefficient `-0.002756`, |coef| `0.002756`
- `lag_00__CT1__flash_duration`: coefficient `0.002546`, |coef| `0.002546`
- `lag_00__CT_place_STAIRS`: coefficient `0.002199`, |coef| `0.002199`
- `lag_00__kill_diff_last_3s`: coefficient `0.002168`, |coef| `0.002168`
- `lag_00__CT_defusing_count`: coefficient `0.002149`, |coef| `0.002149`
- `lag_01__T_place_SCAFFOLDING`: coefficient `-0.002145`, |coef| `0.002145`
- `lag_02__T_place_SCAFFOLDING`: coefficient `-0.002008`, |coef| `0.002008`
- `lag_13__CT_place_SNIPERSNEST`: coefficient `0.001963`, |coef| `0.001963`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001882`, |coef| `0.001882`
- `lag_00__T_damage_last_5s`: coefficient `-0.001796`, |coef| `0.001796`
- `lag_00__damage_diff_last_5s`: coefficient `0.001790`, |coef| `0.001790`
- `lag_09__T_place_SCAFFOLDING`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_13__T_place_PALACEINTERIOR`: coefficient `-0.001631`, |coef| `0.001631`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `0.002546` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001882` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.001450` (lowers CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `-0.001351` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001345` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.001216` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001169` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001155` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.001131` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.001121` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_SCAFFOLDING`: coefficient `-0.003182` (lowers CT win probability)
- `lag_08__T_place_SCAFFOLDING`: coefficient `-0.002920` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002756` (lowers CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `0.002199` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002168` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002149` (raises CT win probability)
- `lag_01__T_place_SCAFFOLDING`: coefficient `-0.002145` (lowers CT win probability)
- `lag_02__T_place_SCAFFOLDING`: coefficient `-0.002008` (lowers CT win probability)
- `lag_13__CT_place_SNIPERSNEST`: coefficient `0.001963` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001796` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `57053`, seconds `103.50`, LSTM delta `-0.4077`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.017461`
- `lag_00__CT_place_STAIRS`: contribution `-0.017113`
- `lag_00__CT1__flash_duration`: contribution `-0.016582`
- `lag_12__T2__is_scoped`: contribution `-0.012861`
- `lag_13__CT_place_SNIPERSNEST`: contribution `-0.010513`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.016582`
- `lag_00__CT_flash_duration_sum`: contribution `-0.008302`
- `lag_05__T2__flash_duration`: contribution `-0.006560`
- `lag_06__CT2__flash_duration`: contribution `-0.005390`
- `lag_05__CT1__flash_duration`: contribution `-0.005193`

### tick `57917`, seconds `117.00`, LSTM delta `+0.2398`

Top all feature movements:
- `lag_14__T_place_SCAFFOLDING`: contribution `+0.108362`
- `lag_00__CT_defusing_count`: contribution `+0.020835`
- `lag_06__T_flash_alpha_mean`: contribution `+0.008161`
- `lag_15__T2__is_scoped`: contribution `+0.005334`
- `lag_00__kill_diff_last_3s`: contribution `-0.005219`

Top utility-only movements:
- `lag_06__T_flash_alpha_mean`: contribution `+0.008161`

### tick `57725`, seconds `114.00`, LSTM delta `+0.1980`

Top all feature movements:
- `lag_08__T_place_SCAFFOLDING`: contribution `+0.099442`
- `lag_08__T2__is_scoped`: contribution `-0.009817`
- `lag_00__T_flash_alpha_mean`: contribution `+0.005890`
- `lag_13__T_place_SCAFFOLDING`: contribution `+0.005860`
- `lag_09__T2__is_scoped`: contribution `+0.005529`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.005890`

### tick `57533`, seconds `111.00`, LSTM delta `+0.1757`

Top all feature movements:
- `lag_02__T_place_SCAFFOLDING`: contribution `+0.068378`
- `lag_07__T_place_SCAFFOLDING`: contribution `+0.013216`
- `lag_00__kill_diff_last_3s`: contribution `+0.010438`
- `lag_00__T_kills_last_3s`: contribution `+0.008730`
- `lag_09__T_kills_last_3s`: contribution `+0.004827`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `+0.004501`
- `lag_15__CT_flash_duration_sum`: contribution `+0.002797`

### tick `56925`, seconds `101.50`, LSTM delta `+0.1523`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `+0.017113`
- `lag_00__CT1__flash_duration`: contribution `+0.010469`
- `lag_08__T2__is_scoped`: contribution `+0.009817`
- `lag_00__kill_diff_last_3s`: contribution `+0.005219`
- `lag_15__CT_place_STAIRS`: contribution `+0.004580`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.010469`
- `lag_01__CT4__flash_duration`: contribution `+0.003934`
- `lag_02__CT2__flash_duration`: contribution `+0.002331`
