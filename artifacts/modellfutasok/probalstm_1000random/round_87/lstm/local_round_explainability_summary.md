# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `20`

## Largest probability jumps

- tick `179364`, seconds `98.50`, LSTM `0.0442`, delta `-0.0762`
- tick `178692`, seconds `88.00`, LSTM `0.2584`, delta `-0.0633`
- tick `178980`, seconds `92.50`, LSTM `0.1653`, delta `-0.0524`
- tick `178436`, seconds `84.00`, LSTM `0.3974`, delta `-0.0518`
- tick `178372`, seconds `83.00`, LSTM `0.4689`, delta `-0.0500`
- tick `178916`, seconds `91.50`, LSTM `0.2292`, delta `-0.0416`
- tick `174244`, seconds `18.50`, LSTM `0.5159`, delta `+0.0360`
- tick `176548`, seconds `54.50`, LSTM `0.5564`, delta `-0.0333`
- tick `178532`, seconds `85.50`, LSTM `0.3227`, delta `-0.0310`
- tick `177188`, seconds `64.50`, LSTM `0.4775`, delta `-0.0291`

## Top 15 local ridge features

- `lag_03__CT1__shots_fired`: coefficient `-0.001029`, |coef| `0.001029`
- `lag_00__CT_place_TSPAWN`: coefficient `-0.001019`, |coef| `0.001019`
- `lag_04__CT1__shots_fired`: coefficient `-0.001003`, |coef| `0.001003`
- `lag_01__CT_place_TSPAWN`: coefficient `-0.000988`, |coef| `0.000988`
- `lag_02__CT1__shots_fired`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_05__CT_place_TSPAWN`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_05__CT1__shots_fired`: coefficient `-0.000945`, |coef| `0.000945`
- `lag_06__CT_place_LOBBY`: coefficient `-0.000942`, |coef| `0.000942`
- `lag_06__CT1__shots_fired`: coefficient `-0.000916`, |coef| `0.000916`
- `lag_02__CT_place_TSPAWN`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_04__CT_place_TSPAWN`: coefficient `-0.000889`, |coef| `0.000889`
- `lag_03__CT_place_TSPAWN`: coefficient `-0.000853`, |coef| `0.000853`
- `lag_06__CT_place_TSPAWN`: coefficient `-0.000836`, |coef| `0.000836`
- `lag_11__CT1__shots_fired`: coefficient `-0.000820`, |coef| `0.000820`
- `lag_07__CT1__shots_fired`: coefficient `-0.000792`, |coef| `0.000792`

## Top 10 utility ridge features

- `lag_03__T_A_site_active_infernos`: coefficient `-0.000488` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000461` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000461` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `0.000340` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.000326` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `0.000323` (raises CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.000306` (lowers CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `0.000299` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000292` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.000290` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT1__shots_fired`: coefficient `-0.001029` (lowers CT win probability)
- `lag_00__CT_place_TSPAWN`: coefficient `-0.001019` (lowers CT win probability)
- `lag_04__CT1__shots_fired`: coefficient `-0.001003` (lowers CT win probability)
- `lag_01__CT_place_TSPAWN`: coefficient `-0.000988` (lowers CT win probability)
- `lag_02__CT1__shots_fired`: coefficient `-0.000979` (lowers CT win probability)
- `lag_05__CT_place_TSPAWN`: coefficient `-0.000974` (lowers CT win probability)
- `lag_05__CT1__shots_fired`: coefficient `-0.000945` (lowers CT win probability)
- `lag_06__CT_place_LOBBY`: coefficient `-0.000942` (lowers CT win probability)
- `lag_06__CT1__shots_fired`: coefficient `-0.000916` (lowers CT win probability)
- `lag_02__CT_place_TSPAWN`: coefficient `-0.000903` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `179364`, seconds `98.50`, LSTM delta `-0.0762`

Top all feature movements:
- `lag_00__CT_place_ROOF`: contribution `-0.014214`
- `lag_06__CT_place_LOBBY`: contribution `-0.007711`
- `lag_12__CT_place_HUT`: contribution `-0.005907`
- `lag_06__CT_place_HUT`: contribution `-0.005675`
- `lag_00__CT_place_LOBBY`: contribution `-0.003095`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `-0.000911`
- `lag_13__T_B_site_active_infernos`: contribution `-0.000815`

### tick `178692`, seconds `88.00`, LSTM delta `-0.0633`

Top all feature movements:
- `lag_04__CT_place_LOCKERROOM`: contribution `-0.009741`
- `lag_10__CT_shots_fired_sum`: contribution `-0.003619`
- `lag_09__T_shots_fired_sum`: contribution `-0.002757`
- `lag_12__CT1__shots_fired`: contribution `-0.002432`
- `lag_11__CT1__shots_fired`: contribution `-0.002167`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `-0.000911`
- `lag_13__T_B_site_active_infernos`: contribution `-0.000815`

### tick `178980`, seconds `92.50`, LSTM delta `-0.0524`

Top all feature movements:
- `lag_00__CT_place_HUT`: contribution `-0.006493`
- `lag_05__CT_place_HUT`: contribution `-0.006086`
- `lag_05__CT_place_LOBBY`: contribution `-0.005993`
- `lag_13__CT_place_LOCKERROOM`: contribution `-0.005036`
- `lag_02__T_bomb_zone_count`: contribution `-0.002305`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `178436`, seconds `84.00`, LSTM delta `-0.0518`

Top all feature movements:
- `lag_04__CT1__shots_fired`: contribution `-0.003178`
- `lag_03__CT1__shots_fired`: contribution `-0.002718`
- `lag_02__CT_shots_fired_sum`: contribution `-0.001792`
- `lag_11__CT_place_HEAVEN`: contribution `-0.001783`
- `lag_09__T3__duck_amount`: contribution `-0.001552`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `178372`, seconds `83.00`, LSTM delta `-0.0500`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.004187`
- `lag_02__CT1__shots_fired`: contribution `-0.003105`
- `lag_00__CT3__duck_amount`: contribution `-0.002631`
- `lag_02__T4__is_scoped`: contribution `-0.002405`
- `lag_06__CT_place_RAFTERS`: contribution `-0.002311`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.001453`
- `lag_03__T_B_site_active_infernos`: contribution `-0.001304`
