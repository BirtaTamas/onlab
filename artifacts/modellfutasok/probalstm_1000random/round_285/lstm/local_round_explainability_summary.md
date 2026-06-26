# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `84063`, seconds `61.00`, LSTM `0.9137`, delta `+0.2444`
- tick `83999`, seconds `60.00`, LSTM `0.6462`, delta `+0.0561`
- tick `83935`, seconds `59.00`, LSTM `0.5740`, delta `-0.0405`
- tick `83871`, seconds `58.00`, LSTM `0.6256`, delta `-0.0345`
- tick `83711`, seconds `55.50`, LSTM `0.6553`, delta `-0.0341`
- tick `84095`, seconds `61.50`, LSTM `0.9410`, delta `+0.0273`
- tick `80703`, seconds `8.50`, LSTM `0.7574`, delta `+0.0240`
- tick `80415`, seconds `4.00`, LSTM `0.7199`, delta `-0.0236`
- tick `84031`, seconds `60.50`, LSTM `0.6693`, delta `+0.0231`
- tick `83551`, seconds `53.00`, LSTM `0.7137`, delta `+0.0227`

## Top 15 local ridge features

- `lag_00__T_place_BDOORS`: coefficient `-0.002499`, |coef| `0.002499`
- `lag_04__T_place_BDOORS`: coefficient `0.001816`, |coef| `0.001816`
- `lag_04__T_place_MIDDOORS`: coefficient `-0.001054`, |coef| `0.001054`
- `lag_06__T_place_BDOORS`: coefficient `0.000878`, |coef| `0.000878`
- `lag_12__T_flashed_players`: coefficient `0.000782`, |coef| `0.000782`
- `lag_02__CT_place_ARAMP`: coefficient `-0.000764`, |coef| `0.000764`
- `lag_14__T1__is_walking`: coefficient `-0.000731`, |coef| `0.000731`
- `lag_12__CT4__flash_duration`: coefficient `0.000724`, |coef| `0.000724`
- `lag_00__CT_place_HOLE`: coefficient `0.000722`, |coef| `0.000722`
- `lag_05__T_place_BDOORS`: coefficient `0.000670`, |coef| `0.000670`
- `lag_01__T_place_BDOORS`: coefficient `-0.000632`, |coef| `0.000632`
- `lag_03__T_place_MIDDOORS`: coefficient `-0.000615`, |coef| `0.000615`
- `lag_00__CT_kills_last_3s`: coefficient `0.000610`, |coef| `0.000610`
- `lag_06__T_flashed_players`: coefficient `-0.000571`, |coef| `0.000571`
- `lag_02__T_place_MIDDOORS`: coefficient `-0.000558`, |coef| `0.000558`

## Top 10 utility ridge features

- `lag_12__CT4__flash_duration`: coefficient `0.000724` (raises CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.000508` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.000488` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000465` (lowers CT win probability)
- `lag_04__CT1__molly`: coefficient `-0.000392` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000388` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.000378` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `0.000378` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000367` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `-0.000363` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BDOORS`: coefficient `-0.002499` (lowers CT win probability)
- `lag_04__T_place_BDOORS`: coefficient `0.001816` (raises CT win probability)
- `lag_04__T_place_MIDDOORS`: coefficient `-0.001054` (lowers CT win probability)
- `lag_06__T_place_BDOORS`: coefficient `0.000878` (raises CT win probability)
- `lag_12__T_flashed_players`: coefficient `0.000782` (raises CT win probability)
- `lag_02__CT_place_ARAMP`: coefficient `-0.000764` (lowers CT win probability)
- `lag_14__T1__is_walking`: coefficient `-0.000731` (lowers CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.000722` (raises CT win probability)
- `lag_05__T_place_BDOORS`: coefficient `0.000670` (raises CT win probability)
- `lag_01__T_place_BDOORS`: coefficient `-0.000632` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `84063`, seconds `61.00`, LSTM delta `+0.2444`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.062507`
- `lag_04__T_place_BDOORS`: contribution `+0.045433`
- `lag_06__T_place_BDOORS`: contribution `+0.010984`
- `lag_04__T_place_MIDDOORS`: contribution `+0.008955`
- `lag_12__T_flashed_players`: contribution `+0.006036`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `+0.004677`
- `lag_03__CT2__flash_duration`: contribution `+0.002391`
- `lag_03__CT4__flash_duration`: contribution `+0.001917`

### tick `83999`, seconds `60.00`, LSTM delta `+0.0561`

Top all feature movements:
- `lag_04__T_place_BDOORS`: contribution `+0.022717`
- `lag_02__T_place_BDOORS`: contribution `+0.012149`
- `lag_01__T_place_BDOORS`: contribution `+0.007901`
- `lag_02__T_place_MIDDOORS`: contribution `+0.004742`
- `lag_04__T_place_MIDDOORS`: contribution `+0.004478`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.001824`
- `lag_10__CT4__flash_duration`: contribution `+0.000922`

### tick `83935`, seconds `59.00`, LSTM delta `-0.0405`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.062507`
- `lag_02__T_place_BDOORS`: contribution `+0.006075`
- `lag_00__T_place_MIDDOORS`: contribution `+0.003975`
- `lag_06__CT2__is_scoped`: contribution `-0.003381`
- `lag_02__T_place_MIDDOORS`: contribution `+0.002371`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83871`, seconds `58.00`, LSTM delta `-0.0345`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.031253`
- `lag_06__T_flashed_players`: contribution `-0.004405`
- `lag_05__CT2__is_scoped`: contribution `+0.002089`
- `lag_00__T_place_MIDDOORS`: contribution `+0.001987`
- `lag_14__T1__is_walking`: contribution `+0.001668`

Top utility-only movements:
- `lag_06__T_flash_duration_sum`: contribution `-0.001409`
- `lag_06__T2__flash_duration`: contribution `-0.001163`
- `lag_06__T1__flash_duration`: contribution `-0.000781`

### tick `83711`, seconds `55.50`, LSTM delta `-0.0341`

Top all feature movements:
- `lag_01__T_flashed_players`: contribution `-0.003474`
- `lag_01__CT4__flash_duration`: contribution `-0.003000`
- `lag_00__CT_place_BDOORS`: contribution `-0.002401`
- `lag_00__T_place_MIDDOORS`: contribution `-0.001987`
- `lag_11__CT2__is_scoped`: contribution `+0.001799`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `-0.003000`
- `lag_01__T_flash_duration_sum`: contribution `-0.001001`
- `lag_01__T2__flash_duration`: contribution `-0.000734`
