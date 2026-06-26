# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `3`

## Largest probability jumps

- tick `30580`, seconds `88.50`, LSTM `0.6709`, delta `+0.2068`
- tick `30644`, seconds `89.50`, LSTM `0.7792`, delta `+0.1633`
- tick `30260`, seconds `83.50`, LSTM `0.6700`, delta `+0.0817`
- tick `30324`, seconds `84.50`, LSTM `0.5485`, delta `-0.0720`
- tick `30708`, seconds `90.50`, LSTM `0.8643`, delta `+0.0719`
- tick `30612`, seconds `89.00`, LSTM `0.6159`, delta `-0.0550`
- tick `30292`, seconds `84.00`, LSTM `0.6205`, delta `-0.0495`
- tick `30740`, seconds `91.00`, LSTM `0.9128`, delta `+0.0484`
- tick `29844`, seconds `77.00`, LSTM `0.6522`, delta `-0.0437`
- tick `29588`, seconds `73.00`, LSTM `0.6962`, delta `-0.0435`

## Top 15 local ridge features

- `lag_10__CT_place_STORAGEROOM`: coefficient `-0.003406`, |coef| `0.003406`
- `lag_12__CT_place_STORAGEROOM`: coefficient `-0.002040`, |coef| `0.002040`
- `lag_14__CT_place_STORAGEROOM`: coefficient `-0.001676`, |coef| `0.001676`
- `lag_08__T_place_CANAL`: coefficient `-0.001512`, |coef| `0.001512`
- `lag_13__CT_place_STORAGEROOM`: coefficient `-0.001145`, |coef| `0.001145`
- `lag_10__T_place_CANAL`: coefficient `-0.001109`, |coef| `0.001109`
- `lag_08__CT_place_STORAGEROOM`: coefficient `0.001068`, |coef| `0.001068`
- `lag_11__T_place_CANAL`: coefficient `-0.001033`, |coef| `0.001033`
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.001017`, |coef| `0.001017`
- `lag_07__T_place_CANAL`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_00__CT_place_STAIRS`: coefficient `-0.000955`, |coef| `0.000955`
- `lag_03__T_place_CANAL`: coefficient `-0.000951`, |coef| `0.000951`
- `lag_12__CT3__is_walking`: coefficient `-0.000950`, |coef| `0.000950`
- `lag_01__CT4__duck_amount`: coefficient `-0.000935`, |coef| `0.000935`
- `lag_04__CT_place_STORAGEROOM`: coefficient `-0.000916`, |coef| `0.000916`

## Top 10 utility ridge features

- `lag_01__CT_B_site_active_infernos`: coefficient `0.000462` (raises CT win probability)
- `lag_04__CT1__molly`: coefficient `-0.000376` (lowers CT win probability)
- `lag_06__CT1__molly`: coefficient `-0.000342` (lowers CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.000317` (raises CT win probability)
- `lag_11__CT_B_site_active_smokes`: coefficient `-0.000290` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000263` (lowers CT win probability)
- `lag_07__T1__smoke`: coefficient `-0.000257` (lowers CT win probability)
- `lag_14__CT2__smoke`: coefficient `0.000255` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000252` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000241` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_STORAGEROOM`: coefficient `-0.003406` (lowers CT win probability)
- `lag_12__CT_place_STORAGEROOM`: coefficient `-0.002040` (lowers CT win probability)
- `lag_14__CT_place_STORAGEROOM`: coefficient `-0.001676` (lowers CT win probability)
- `lag_08__T_place_CANAL`: coefficient `-0.001512` (lowers CT win probability)
- `lag_13__CT_place_STORAGEROOM`: coefficient `-0.001145` (lowers CT win probability)
- `lag_10__T_place_CANAL`: coefficient `-0.001109` (lowers CT win probability)
- `lag_08__CT_place_STORAGEROOM`: coefficient `0.001068` (raises CT win probability)
- `lag_11__T_place_CANAL`: coefficient `-0.001033` (lowers CT win probability)
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.001017` (lowers CT win probability)
- `lag_07__T_place_CANAL`: coefficient `-0.000962` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `30580`, seconds `88.50`, LSTM delta `+0.2068`

Top all feature movements:
- `lag_10__CT_place_STORAGEROOM`: contribution `+0.072862`
- `lag_08__T_place_CANAL`: contribution `+0.008408`
- `lag_01__CT_place_STAIRS`: contribution `+0.005611`
- `lag_04__T_bomb_zone_count`: contribution `+0.004193`
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.004156`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30644`, seconds `89.50`, LSTM delta `+0.1633`

Top all feature movements:
- `lag_12__CT_place_STORAGEROOM`: contribution `+0.043647`
- `lag_00__CT_place_STAIRS`: contribution `+0.007436`
- `lag_10__T_place_CANAL`: contribution `+0.006166`
- `lag_03__CT_place_STAIRS`: contribution `+0.005318`
- `lag_06__T_bomb_zone_count`: contribution `+0.003775`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30260`, seconds `83.50`, LSTM delta `+0.0817`

Top all feature movements:
- `lag_08__CT_place_STORAGEROOM`: contribution `+0.022843`
- `lag_00__CT_place_STORAGEROOM`: contribution `+0.014772`
- `lag_08__CT_place_LOBBY`: contribution `+0.005959`
- `lag_12__CT3__is_walking`: contribution `+0.002268`
- `lag_06__T_place_CANAL`: contribution `-0.002231`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30324`, seconds `84.50`, LSTM delta `-0.0720`

Top all feature movements:
- `lag_10__CT_place_STORAGEROOM`: contribution `-0.072862`
- `lag_08__T_place_CANAL`: contribution `-0.004204`
- `lag_02__CT_place_STORAGEROOM`: contribution `+0.004083`
- `lag_10__CT_place_LOBBY`: contribution `+0.002836`
- `lag_00__T_macro_B`: contribution `-0.002083`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30708`, seconds `90.50`, LSTM delta `+0.0719`

Top all feature movements:
- `lag_14__CT_place_STORAGEROOM`: contribution `+0.035845`
- `lag_08__T_place_CANAL`: contribution `+0.004204`
- `lag_12__T_place_CANAL`: contribution `+0.003632`
- `lag_05__CT_place_BACKOFA`: contribution `-0.002881`
- `lag_07__T_place_CANAL`: contribution `+0.002676`

Top utility-only movements:
- No utility movement among the top local contributors.
