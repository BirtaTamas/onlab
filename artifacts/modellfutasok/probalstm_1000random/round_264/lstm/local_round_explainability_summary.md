# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `143460`, seconds `20.50`, LSTM `0.1485`, delta `-0.3774`
- tick `143428`, seconds `20.00`, LSTM `0.5259`, delta `-0.0763`
- tick `143492`, seconds `21.00`, LSTM `0.0897`, delta `-0.0588`
- tick `144004`, seconds `29.00`, LSTM `0.0077`, delta `-0.0369`
- tick `142756`, seconds `9.50`, LSTM `0.6251`, delta `+0.0249`
- tick `143076`, seconds `14.50`, LSTM `0.6109`, delta `-0.0217`
- tick `143556`, seconds `22.00`, LSTM `0.0526`, delta `-0.0187`
- tick `143524`, seconds `21.50`, LSTM `0.0712`, delta `-0.0185`
- tick `143204`, seconds `16.50`, LSTM `0.5857`, delta `-0.0157`
- tick `143844`, seconds `26.50`, LSTM `0.0335`, delta `-0.0150`

## Top 15 local ridge features

- `lag_02__CT2__is_scoped`: coefficient `0.002638`, |coef| `0.002638`
- `lag_01__CT_place_TRUCK`: coefficient `0.002540`, |coef| `0.002540`
- `lag_14__CT_place_TRUCK`: coefficient `-0.002450`, |coef| `0.002450`
- `lag_07__CT_place_JUNGLE`: coefficient `-0.002302`, |coef| `0.002302`
- `lag_13__CT_place_JUNGLE`: coefficient `-0.002193`, |coef| `0.002193`
- `lag_12__CT_place_TRUCK`: coefficient `0.002029`, |coef| `0.002029`
- `lag_03__T_burning_players`: coefficient `-0.001903`, |coef| `0.001903`
- `lag_00__T_kills_last_3s`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_11__CT_place_JUNGLE`: coefficient `0.001882`, |coef| `0.001882`
- `lag_01__T_kills_last_3s`: coefficient `-0.001685`, |coef| `0.001685`
- `lag_12__CT_place_APARTMENTS`: coefficient `-0.001620`, |coef| `0.001620`
- `lag_15__CT_A_site_active_infernos`: coefficient `0.001564`, |coef| `0.001564`
- `lag_00__T_damage_last_5s`: coefficient `-0.001531`, |coef| `0.001531`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001526`, |coef| `0.001526`
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.001506`, |coef| `0.001506`

## Top 10 utility ridge features

- `lag_15__CT_A_site_active_infernos`: coefficient `0.001564` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.001506` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001296` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001286` (raises CT win probability)
- `lag_08__CT3__molly`: coefficient `0.001165` (raises CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.001083` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001082` (raises CT win probability)
- `lag_08__CT4__smoke`: coefficient `0.001059` (raises CT win probability)
- `lag_01__CT2__utility_total`: coefficient `0.001000` (raises CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.000934` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT2__is_scoped`: coefficient `0.002638` (raises CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `0.002540` (raises CT win probability)
- `lag_14__CT_place_TRUCK`: coefficient `-0.002450` (lowers CT win probability)
- `lag_07__CT_place_JUNGLE`: coefficient `-0.002302` (lowers CT win probability)
- `lag_13__CT_place_JUNGLE`: coefficient `-0.002193` (lowers CT win probability)
- `lag_12__CT_place_TRUCK`: coefficient `0.002029` (raises CT win probability)
- `lag_03__T_burning_players`: coefficient `-0.001903` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001899` (lowers CT win probability)
- `lag_11__CT_place_JUNGLE`: coefficient `0.001882` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001685` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `143460`, seconds `20.50`, LSTM delta `-0.3774`

Top all feature movements:
- `lag_01__CT_place_TRUCK`: contribution `-0.016386`
- `lag_02__CT2__is_scoped`: contribution `-0.016147`
- `lag_14__CT_place_TRUCK`: contribution `-0.015801`
- `lag_07__CT_place_JUNGLE`: contribution `-0.014765`
- `lag_13__CT_place_JUNGLE`: contribution `-0.014067`

Top utility-only movements:
- `lag_15__CT_A_site_active_infernos`: contribution `-0.005521`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.005173`

### tick `143428`, seconds `20.00`, LSTM delta `-0.0763`

Top all feature movements:
- `lag_11__CT_place_TRUCK`: contribution `-0.006816`
- `lag_00__CT_place_TRUCK`: contribution `-0.006097`
- `lag_00__T_kills_last_3s`: contribution `-0.006017`
- `lag_06__CT_place_JUNGLE`: contribution `-0.005383`
- `lag_01__CT2__is_scoped`: contribution `-0.004363`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `143492`, seconds `21.00`, LSTM delta `-0.0588`

Top all feature movements:
- `lag_15__CT_place_TRUCK`: contribution `-0.007058`
- `lag_08__CT_place_JUNGLE`: contribution `-0.006196`
- `lag_01__T_kills_last_3s`: contribution `-0.005339`
- `lag_01__CT3__duck_amount`: contribution `+0.004280`
- `lag_15__CT3__duck_amount`: contribution `+0.003677`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `144004`, seconds `29.00`, LSTM delta `-0.0369`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006017`
- `lag_00__kill_diff_last_3s`: contribution `-0.003473`
- `lag_00__CT1__is_scoped`: contribution `-0.003049`
- `lag_14__CT_place_JUNGLE`: contribution `+0.002840`
- `lag_03__CT5__is_walking`: contribution `-0.002426`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `142756`, seconds `9.50`, LSTM delta `+0.0249`

Top all feature movements:
- `lag_12__CT1__duck_amount`: contribution `+0.004578`
- `lag_03__T_place_HOUSE`: contribution `-0.004371`
- `lag_09__T_place_HOUSE`: contribution `+0.003981`
- `lag_00__CT1__duck_amount`: contribution `+0.003895`
- `lag_03__T_place_BACKALLEY`: contribution `-0.003523`

Top utility-only movements:
- `lag_04__CT_active_infernos`: contribution `-0.002002`
