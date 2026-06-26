# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `27`

## Largest probability jumps

- tick `212640`, seconds `111.00`, LSTM `0.3424`, delta `-0.2771`
- tick `211712`, seconds `96.50`, LSTM `0.6641`, delta `+0.1791`
- tick `211040`, seconds `86.00`, LSTM `0.5836`, delta `+0.0994`
- tick `212032`, seconds `101.50`, LSTM `0.6656`, delta `-0.0813`
- tick `212768`, seconds `113.00`, LSTM `0.3692`, delta `+0.0775`
- tick `211008`, seconds `85.50`, LSTM `0.4842`, delta `-0.0696`
- tick `212576`, seconds `110.00`, LSTM `0.6210`, delta `+0.0676`
- tick `211936`, seconds `100.00`, LSTM `0.7361`, delta `+0.0635`
- tick `212320`, seconds `106.00`, LSTM `0.6441`, delta `+0.0566`
- tick `211744`, seconds `97.00`, LSTM `0.7185`, delta `+0.0544`

## Top 15 local ridge features

- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.003132`, |coef| `0.003132`
- `lag_05__T_place_SIDEHALL`: coefficient `0.002643`, |coef| `0.002643`
- `lag_00__kill_diff_last_3s`: coefficient `0.002563`, |coef| `0.002563`
- `lag_08__T_bomb_zone_count`: coefficient `-0.002545`, |coef| `0.002545`
- `lag_04__CT2__duck_amount`: coefficient `0.002410`, |coef| `0.002410`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.002096`, |coef| `0.002096`
- `lag_02__T_bomb_zone_count`: coefficient `0.001974`, |coef| `0.001974`
- `lag_04__CT_duck_amount_mean`: coefficient `0.001931`, |coef| `0.001931`
- `lag_07__CT2__duck_amount`: coefficient `-0.001928`, |coef| `0.001928`
- `lag_00__damage_diff_last_5s`: coefficient `0.001883`, |coef| `0.001883`
- `lag_00__T_kills_last_3s`: coefficient `-0.001620`, |coef| `0.001620`
- `lag_00__CT_kills_last_3s`: coefficient `0.001597`, |coef| `0.001597`
- `lag_05__CT2__is_scoped`: coefficient `0.001567`, |coef| `0.001567`
- `lag_08__T_place_SIDEHALL`: coefficient `0.001558`, |coef| `0.001558`
- `lag_15__T5__is_walking`: coefficient `0.001465`, |coef| `0.001465`

## Top 10 utility ridge features

- `lag_12__CT_A_site_active_infernos`: coefficient `-0.001455` (lowers CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.001250` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001083` (raises CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.001054` (lowers CT win probability)
- `lag_13__CT3__molly`: coefficient `0.000961` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000808` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000763` (raises CT win probability)
- `lag_10__CT5__molly`: coefficient `-0.000726` (lowers CT win probability)
- `lag_05__CT4__flash`: coefficient `0.000659` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000607` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.003132` (raises CT win probability)
- `lag_05__T_place_SIDEHALL`: coefficient `0.002643` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002563` (raises CT win probability)
- `lag_08__T_bomb_zone_count`: coefficient `-0.002545` (lowers CT win probability)
- `lag_04__CT2__duck_amount`: coefficient `0.002410` (raises CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.002096` (lowers CT win probability)
- `lag_02__T_bomb_zone_count`: coefficient `0.001974` (raises CT win probability)
- `lag_04__CT_duck_amount_mean`: coefficient `0.001931` (raises CT win probability)
- `lag_07__CT2__duck_amount`: coefficient `-0.001928` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001883` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `212640`, seconds `111.00`, LSTM delta `-0.2771`

Top all feature movements:
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.023543`
- `lag_08__T_bomb_zone_count`: contribution `-0.014817`
- `lag_02__T_bomb_zone_count`: contribution `-0.011489`
- `lag_04__CT2__duck_amount`: contribution `-0.009183`
- `lag_04__CT_duck_amount_mean`: contribution `-0.008828`

Top utility-only movements:
- `lag_12__CT_A_site_active_infernos`: contribution `-0.005135`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.004411`

### tick `211712`, seconds `96.50`, LSTM delta `+0.1791`

Top all feature movements:
- `lag_05__T_place_SIDEHALL`: contribution `+0.017130`
- `lag_00__T_place_SIDEHALL`: contribution `+0.013585`
- `lag_07__T_place_SIDEHALL`: contribution `+0.008204`
- `lag_07__CT2__duck_amount`: contribution `+0.007347`
- `lag_01__T_place_SIDEHALL`: contribution `+0.006957`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `+0.003722`

### tick `211040`, seconds `86.00`, LSTM delta `+0.0994`

Top all feature movements:
- `lag_05__CT2__is_scoped`: contribution `+0.009593`
- `lag_04__CT2__duck_amount`: contribution `+0.009183`
- `lag_00__kill_diff_last_3s`: contribution `+0.006168`
- `lag_00__CT_kills_last_3s`: contribution `+0.004611`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.004031`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `212032`, seconds `101.50`, LSTM delta `-0.0813`

Top all feature movements:
- `lag_05__T_place_SIDEHALL`: contribution `-0.017130`
- `lag_04__CT2__duck_amount`: contribution `-0.009183`
- `lag_00__T_bomb_zone_count`: contribution `-0.008125`
- `lag_07__CT2__duck_amount`: contribution `-0.007347`
- `lag_07__CT2__is_scoped`: contribution `-0.007303`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `212768`, seconds `113.00`, LSTM delta `+0.0775`

Top all feature movements:
- `lag_07__CT2__is_scoped`: contribution `+0.007303`
- `lag_03__CT2__is_scoped`: contribution `+0.006627`
- `lag_01__T4__is_scoped`: contribution `+0.005668`
- `lag_01__T2__shots_fired`: contribution `+0.005426`
- `lag_00__CT2__is_scoped`: contribution `+0.004450`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001549`
