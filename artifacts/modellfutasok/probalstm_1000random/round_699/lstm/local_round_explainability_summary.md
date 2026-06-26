# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `35934`, seconds `0.50`, LSTM `0.0436`, delta `-0.0455`
- tick `37822`, seconds `30.00`, LSTM `0.0505`, delta `-0.0399`
- tick `38110`, seconds `34.50`, LSTM `0.0170`, delta `-0.0305`
- tick `37054`, seconds `18.00`, LSTM `0.1046`, delta `+0.0189`
- tick `36670`, seconds `12.00`, LSTM `0.0701`, delta `-0.0180`
- tick `37854`, seconds `30.50`, LSTM `0.0337`, delta `-0.0168`
- tick `36638`, seconds `11.50`, LSTM `0.0881`, delta `+0.0153`
- tick `36702`, seconds `12.50`, LSTM `0.0848`, delta `+0.0147`
- tick `37406`, seconds `23.50`, LSTM `0.1101`, delta `+0.0126`
- tick `38046`, seconds `33.50`, LSTM `0.0521`, delta `+0.0099`

## Top 15 local ridge features

- `lag_12__CT_place_SHOP`: coefficient `0.000499`, |coef| `0.000499`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000471`, |coef| `0.000471`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000433`, |coef| `0.000433`
- `lag_00__CT_velocity_mean`: coefficient `-0.000391`, |coef| `0.000391`
- `lag_11__CT_place_SHOP`: coefficient `0.000386`, |coef| `0.000386`
- `lag_13__CT_place_JUNGLE`: coefficient `-0.000336`, |coef| `0.000336`
- `lag_00__CT1__flash_duration`: coefficient `-0.000329`, |coef| `0.000329`
- `lag_01__T2__duck_amount`: coefficient `0.000302`, |coef| `0.000302`
- `lag_00__T_velocity_mean`: coefficient `-0.000296`, |coef| `0.000296`
- `lag_00__CT_place_JUNGLE`: coefficient `0.000295`, |coef| `0.000295`
- `lag_00__T_kills_last_3s`: coefficient `-0.000291`, |coef| `0.000291`
- `lag_03__CT_place_SHOP`: coefficient `0.000288`, |coef| `0.000288`
- `lag_13__T4__duck_amount`: coefficient `-0.000281`, |coef| `0.000281`
- `lag_09__CT_place_JUNGLE`: coefficient `0.000281`, |coef| `0.000281`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000273`, |coef| `0.000273`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `-0.000329` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000253` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000250` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000243` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `0.000223` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000210` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.000202` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000197` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000182` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000177` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_SHOP`: coefficient `0.000499` (raises CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000471` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000433` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000391` (lowers CT win probability)
- `lag_11__CT_place_SHOP`: coefficient `0.000386` (raises CT win probability)
- `lag_13__CT_place_JUNGLE`: coefficient `-0.000336` (lowers CT win probability)
- `lag_01__T2__duck_amount`: coefficient `0.000302` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000296` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.000295` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000291` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `35934`, seconds `0.50`, LSTM delta `-0.0455`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002251`
- `lag_01__T_place_TSPAWN`: contribution `-0.001916`
- `lag_00__CT_velocity_mean`: contribution `-0.001373`
- `lag_00__T_velocity_mean`: contribution `-0.000834`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000750`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000714`
- `lag_01__smoke_inv_diff`: contribution `-0.000619`
- `lag_01__molly_inv_diff`: contribution `-0.000586`
- `lag_01__T1__utility_total`: contribution `-0.000445`
- `lag_01__T_molly_inv`: contribution `-0.000344`

### tick `37822`, seconds `30.00`, LSTM delta `-0.0399`

Top all feature movements:
- `lag_12__CT_place_SHOP`: contribution `-0.002502`
- `lag_13__CT_place_JUNGLE`: contribution `-0.002156`
- `lag_11__CT_place_SHOP`: contribution `-0.001934`
- `lag_09__CT_place_JUNGLE`: contribution `-0.001802`
- `lag_13__T4__duck_amount`: contribution `-0.001040`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.000752`
- `lag_00__T2__molly`: contribution `-0.000496`

### tick `38110`, seconds `34.50`, LSTM delta `-0.0305`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `-0.002259`
- `lag_07__CT_place_STAIRS`: contribution `-0.001461`
- `lag_03__T1__flash_duration`: contribution `-0.001347`
- `lag_01__T2__duck_amount`: contribution `-0.001155`
- `lag_03__T4__flash_duration`: contribution `-0.000965`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.002259`
- `lag_03__T1__flash_duration`: contribution `-0.001347`
- `lag_03__T4__flash_duration`: contribution `-0.000965`
- `lag_03__T_flash_duration_sum`: contribution `-0.000756`
- `lag_00__CT5__flash_duration`: contribution `-0.000660`

### tick `37054`, seconds `18.00`, LSTM delta `+0.0189`

Top all feature movements:
- `lag_12__CT_place_JUNGLE`: contribution `+0.001428`
- `lag_13__CT_place_TRUCK`: contribution `+0.001326`
- `lag_01__T2__duck_amount`: contribution `+0.001155`
- `lag_09__CT_place_APARTMENTS`: contribution `+0.000748`
- `lag_09__CT_place_TRUCK`: contribution `+0.000640`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36670`, seconds `12.00`, LSTM delta `-0.0180`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `-0.001890`
- `lag_13__T_place_PALACEALLEY`: contribution `-0.001175`
- `lag_10__CT4__duck_amount`: contribution `-0.000959`
- `lag_13__CT_place_SHOP`: contribution `+0.000949`
- `lag_15__T4__duck_amount`: contribution `-0.000709`

Top utility-only movements:
- No utility movement among the top local contributors.
