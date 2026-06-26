# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `25448`, seconds `19.00`, LSTM `0.1036`, delta `-0.2374`
- tick `25416`, seconds `18.50`, LSTM `0.3410`, delta `-0.0692`
- tick `25000`, seconds `12.00`, LSTM `0.4159`, delta `+0.0591`
- tick `25160`, seconds `14.50`, LSTM `0.3629`, delta `+0.0587`
- tick `25576`, seconds `21.00`, LSTM `0.0874`, delta `-0.0582`
- tick `25064`, seconds `13.00`, LSTM `0.3437`, delta `-0.0554`
- tick `24968`, seconds `11.50`, LSTM `0.3567`, delta `-0.0548`
- tick `24840`, seconds `9.50`, LSTM `0.3698`, delta `+0.0547`
- tick `24264`, seconds `0.50`, LSTM `0.2819`, delta `-0.0439`
- tick `24872`, seconds `10.00`, LSTM `0.4129`, delta `+0.0431`

## Top 15 local ridge features

- `lag_12__CT_shots_fired_sum`: coefficient `0.001175`, |coef| `0.001175`
- `lag_02__T_place_DECON`: coefficient `0.001147`, |coef| `0.001147`
- `lag_04__CT_place_HEAVEN`: coefficient `0.000938`, |coef| `0.000938`
- `lag_10__T_place_VENTS`: coefficient `-0.000918`, |coef| `0.000918`
- `lag_00__T_place_SQUEAKY`: coefficient `0.000916`, |coef| `0.000916`
- `lag_12__CT2__shots_fired`: coefficient `0.000873`, |coef| `0.000873`
- `lag_15__T_place_SQUEAKY`: coefficient `0.000771`, |coef| `0.000771`
- `lag_01__CT_place_SQUEAKY`: coefficient `-0.000726`, |coef| `0.000726`
- `lag_03__T_place_VENTS`: coefficient `0.000712`, |coef| `0.000712`
- `lag_04__T_place_DECON`: coefficient `-0.000703`, |coef| `0.000703`
- `lag_15__T_burning_players`: coefficient `-0.000694`, |coef| `0.000694`
- `lag_00__T_place_DECON`: coefficient `0.000654`, |coef| `0.000654`
- `lag_08__T_place_VENTS`: coefficient `0.000641`, |coef| `0.000641`
- `lag_06__CT_place_HEAVEN`: coefficient `0.000627`, |coef| `0.000627`
- `lag_02__CT_place_HELL`: coefficient `0.000623`, |coef| `0.000623`

## Top 10 utility ridge features

- `lag_02__CT2__flash_duration`: coefficient `0.000522` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.000495` (raises CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.000368` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000347` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000310` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.000289` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.000262` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.000242` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `0.000229` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `0.000221` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_shots_fired_sum`: coefficient `0.001175` (raises CT win probability)
- `lag_02__T_place_DECON`: coefficient `0.001147` (raises CT win probability)
- `lag_04__CT_place_HEAVEN`: coefficient `0.000938` (raises CT win probability)
- `lag_10__T_place_VENTS`: coefficient `-0.000918` (lowers CT win probability)
- `lag_00__T_place_SQUEAKY`: coefficient `0.000916` (raises CT win probability)
- `lag_12__CT2__shots_fired`: coefficient `0.000873` (raises CT win probability)
- `lag_15__T_place_SQUEAKY`: coefficient `0.000771` (raises CT win probability)
- `lag_01__CT_place_SQUEAKY`: coefficient `-0.000726` (lowers CT win probability)
- `lag_03__T_place_VENTS`: coefficient `0.000712` (raises CT win probability)
- `lag_04__T_place_DECON`: coefficient `-0.000703` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `25448`, seconds `19.00`, LSTM delta `-0.2374`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `-0.023666`
- `lag_02__T_place_DECON`: contribution `-0.018421`
- `lag_12__CT2__shots_fired`: contribution `-0.013025`
- `lag_10__T_place_VENTS`: contribution `-0.012385`
- `lag_04__T_place_DECON`: contribution `-0.011299`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.003888`

### tick `25416`, seconds `18.50`, LSTM delta `-0.0692`

Top all feature movements:
- `lag_10__T_place_VENTS`: contribution `-0.012385`
- `lag_04__T_place_DECON`: contribution `-0.011299`
- `lag_08__T_place_VENTS`: contribution `-0.008640`
- `lag_12__CT_shots_fired_sum`: contribution `+0.004896`
- `lag_15__T_place_SQUEAKY`: contribution `-0.004803`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.003692`

### tick `25000`, seconds `12.00`, LSTM delta `+0.0591`

Top all feature movements:
- `lag_05__T_place_SQUEAKY`: contribution `+0.006881`
- `lag_04__CT_place_HEAVEN`: contribution `+0.005063`
- `lag_07__CT_place_RAFTERS`: contribution `+0.004930`
- `lag_12__CT_place_HELL`: contribution `+0.004290`
- `lag_13__CT_place_HELL`: contribution `+0.003913`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.002925`

### tick `25160`, seconds `14.50`, LSTM delta `+0.0587`

Top all feature movements:
- `lag_04__T_place_VENTS`: contribution `+0.005823`
- `lag_03__CT_shots_fired_sum`: contribution `+0.005638`
- `lag_00__T_place_VENTS`: contribution `+0.005416`
- `lag_10__T_place_SQUEAKY`: contribution `+0.005289`
- `lag_03__CT2__shots_fired`: contribution `+0.003031`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `+0.002174`

### tick `25576`, seconds `21.00`, LSTM delta `-0.0582`

Top all feature movements:
- `lag_04__T_place_DECON`: contribution `+0.011299`
- `lag_00__CT_place_VENDING`: contribution `-0.008024`
- `lag_06__T_place_DECON`: contribution `-0.007597`
- `lag_13__T_place_VENTS`: contribution `+0.007108`
- `lag_15__T_place_VENTS`: contribution `-0.005957`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `-0.002745`
