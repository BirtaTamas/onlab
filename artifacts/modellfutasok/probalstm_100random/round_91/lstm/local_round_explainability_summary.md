# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `109260`, seconds `70.00`, LSTM `0.0753`, delta `-0.4241`
- tick `108908`, seconds `64.50`, LSTM `0.3870`, delta `-0.1633`
- tick `108492`, seconds `58.00`, LSTM `0.7308`, delta `+0.1068`
- tick `108812`, seconds `63.00`, LSTM `0.6252`, delta `+0.0977`
- tick `108076`, seconds `51.50`, LSTM `0.6102`, delta `+0.0856`
- tick `108844`, seconds `63.50`, LSTM `0.5409`, delta `-0.0843`
- tick `108780`, seconds `62.50`, LSTM `0.5274`, delta `-0.0840`
- tick `109228`, seconds `69.50`, LSTM `0.4994`, delta `+0.0554`
- tick `108524`, seconds `58.50`, LSTM `0.7840`, delta `+0.0532`
- tick `108684`, seconds `61.00`, LSTM `0.6416`, delta `-0.0501`

## Top 15 local ridge features

- `lag_15__CT_place_SILO`: coefficient `0.003147`, |coef| `0.003147`
- `lag_10__T_place_OBSERVATION`: coefficient `0.002129`, |coef| `0.002129`
- `lag_12__T_place_DECON`: coefficient `0.001641`, |coef| `0.001641`
- `lag_05__CT_place_VENTS`: coefficient `-0.001550`, |coef| `0.001550`
- `lag_02__T_place_DECON`: coefficient `0.001343`, |coef| `0.001343`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001314`, |coef| `0.001314`
- `lag_09__CT_place_TROPHY`: coefficient `0.001296`, |coef| `0.001296`
- `lag_15__T_place_DECON`: coefficient `-0.001249`, |coef| `0.001249`
- `lag_00__CT_place_SILO`: coefficient `0.001235`, |coef| `0.001235`
- `lag_10__T_place_DECON`: coefficient `-0.001209`, |coef| `0.001209`
- `lag_11__CT_place_VENDING`: coefficient `0.001203`, |coef| `0.001203`
- `lag_04__CT_place_SILO`: coefficient `0.001160`, |coef| `0.001160`
- `lag_15__CT_place_ROOF`: coefficient `0.001097`, |coef| `0.001097`
- `lag_00__CT_place_VENTS`: coefficient `0.001096`, |coef| `0.001096`
- `lag_08__T_bomb_zone_count`: coefficient `-0.001071`, |coef| `0.001071`

## Top 10 utility ridge features

- `lag_00__CT4__molly`: coefficient `0.000507` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000442` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000410` (lowers CT win probability)
- `lag_14__T_active_infernos`: coefficient `-0.000394` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000390` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.000361` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `-0.000341` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000331` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000322` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000315` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SILO`: coefficient `0.003147` (raises CT win probability)
- `lag_10__T_place_OBSERVATION`: coefficient `0.002129` (raises CT win probability)
- `lag_12__T_place_DECON`: coefficient `0.001641` (raises CT win probability)
- `lag_05__CT_place_VENTS`: coefficient `-0.001550` (lowers CT win probability)
- `lag_02__T_place_DECON`: coefficient `0.001343` (raises CT win probability)
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001314` (lowers CT win probability)
- `lag_09__CT_place_TROPHY`: coefficient `0.001296` (raises CT win probability)
- `lag_15__T_place_DECON`: coefficient `-0.001249` (lowers CT win probability)
- `lag_00__CT_place_SILO`: coefficient `0.001235` (raises CT win probability)
- `lag_10__T_place_DECON`: coefficient `-0.001209` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `109260`, seconds `70.00`, LSTM delta `-0.4241`

Top all feature movements:
- `lag_15__CT_place_SILO`: contribution `-0.210483`
- `lag_10__T_place_OBSERVATION`: contribution `-0.036052`
- `lag_12__T_place_DECON`: contribution `-0.026366`
- `lag_05__CT_place_VENTS`: contribution `-0.013003`
- `lag_00__CT_place_VENTS`: contribution `-0.009199`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108908`, seconds `64.50`, LSTM delta `-0.1633`

Top all feature movements:
- `lag_04__CT_place_SILO`: contribution `-0.077558`
- `lag_15__T_place_DECON`: contribution `-0.020067`
- `lag_01__T_place_DECON`: contribution `-0.007099`
- `lag_13__T_place_DECON`: contribution `-0.005886`
- `lag_11__T_place_OBSERVATION`: contribution `-0.005736`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108492`, seconds `58.00`, LSTM delta `+0.1068`

Top all feature movements:
- `lag_13__CT_place_SILO`: contribution `+0.027954`
- `lag_02__T_place_DECON`: contribution `+0.021570`
- `lag_09__CT_place_TROPHY`: contribution `+0.019137`
- `lag_13__CT_place_OBSERVATION`: contribution `+0.011805`
- `lag_00__T_place_DECON`: contribution `+0.007604`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108812`, seconds `63.00`, LSTM delta `+0.0977`

Top all feature movements:
- `lag_12__T_place_DECON`: contribution `+0.026366`
- `lag_01__CT_place_SILO`: contribution `+0.023132`
- `lag_10__T_place_DECON`: contribution `+0.019426`
- `lag_14__CT_place_CONTROL`: contribution `+0.003400`
- `lag_04__CT_place_ADMIN`: contribution `+0.003091`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `108076`, seconds `51.50`, LSTM delta `+0.0856`

Top all feature movements:
- `lag_00__CT_place_SILO`: contribution `+0.082586`
- `lag_13__CT_place_ROOF`: contribution `-0.003766`
- `lag_09__CT_place_LOBBY`: contribution `+0.002664`
- `lag_00__CT_place_ROOF`: contribution `+0.001735`
- `lag_01__CT4__is_walking`: contribution `+0.001468`

Top utility-only movements:
- No utility movement among the top local contributors.
