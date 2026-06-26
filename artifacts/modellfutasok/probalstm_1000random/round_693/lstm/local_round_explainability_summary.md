# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `18`

## Largest probability jumps

- tick `137599`, seconds `35.50`, LSTM `0.0488`, delta `-0.2847`
- tick `137535`, seconds `34.50`, LSTM `0.3234`, delta `-0.1149`
- tick `136831`, seconds `23.50`, LSTM `0.3122`, delta `-0.0794`
- tick `137503`, seconds `34.00`, LSTM `0.4383`, delta `+0.0719`
- tick `136607`, seconds `20.00`, LSTM `0.2876`, delta `+0.0526`
- tick `136703`, seconds `21.50`, LSTM `0.3640`, delta `+0.0430`
- tick `136863`, seconds `24.00`, LSTM `0.2744`, delta `-0.0378`
- tick `137375`, seconds `32.00`, LSTM `0.3875`, delta `+0.0347`
- tick `136255`, seconds `14.50`, LSTM `0.3477`, delta `-0.0346`
- tick `137439`, seconds `33.00`, LSTM `0.3414`, delta `-0.0338`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002079`, |coef| `0.002079`
- `lag_00__CT2__flash_duration`: coefficient `0.001918`, |coef| `0.001918`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001639`, |coef| `0.001639`
- `lag_00__T_kills_last_3s`: coefficient `-0.001620`, |coef| `0.001620`
- `lag_00__CT_place_QUAD`: coefficient `0.001540`, |coef| `0.001540`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001511`, |coef| `0.001511`
- `lag_00__CT2__shots_fired`: coefficient `0.001487`, |coef| `0.001487`
- `lag_00__CT3__flash_duration`: coefficient `0.001431`, |coef| `0.001431`
- `lag_00__CT_flashed_players`: coefficient `0.001394`, |coef| `0.001394`
- `lag_04__CT_place_QUAD`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_00__T_damage_last_5s`: coefficient `-0.001319`, |coef| `0.001319`
- `lag_05__CT2__flash_duration`: coefficient `-0.001314`, |coef| `0.001314`
- `lag_12__CT_place_PIT`: coefficient `-0.001214`, |coef| `0.001214`
- `lag_01__CT3__is_walking`: coefficient `0.001214`, |coef| `0.001214`
- `lag_00__CT_place_ARCH`: coefficient `0.001204`, |coef| `0.001204`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.001918` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001511` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001431` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.001314` (lowers CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `-0.001178` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.001147` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000878` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000844` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000753` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000747` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002079` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001639` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001620` (lowers CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.001540` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.001487` (raises CT win probability)
- `lag_00__CT_flashed_players`: coefficient `0.001394` (raises CT win probability)
- `lag_04__CT_place_QUAD`: coefficient `-0.001327` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001319` (lowers CT win probability)
- `lag_12__CT_place_PIT`: coefficient `-0.001214` (lowers CT win probability)
- `lag_01__CT3__is_walking`: coefficient `0.001214` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `137599`, seconds `35.50`, LSTM delta `-0.2847`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.014801`
- `lag_00__CT2__flash_duration`: contribution `-0.013884`
- `lag_00__T_kills_last_3s`: contribution `-0.010264`
- `lag_00__CT_flash_duration_sum`: contribution `-0.007913`
- `lag_02__CT2__shots_fired`: contribution `-0.007679`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.013884`
- `lag_00__CT_flash_duration_sum`: contribution `-0.007913`
- `lag_00__CT3__flash_duration`: contribution `-0.006238`
- `lag_07__CT3__flash_duration`: contribution `-0.005533`
- `lag_05__CT2__flash_duration`: contribution `-0.005258`

### tick `137535`, seconds `34.50`, LSTM delta `-0.1149`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.018776`
- `lag_00__CT2__shots_fired`: contribution `-0.009612`
- `lag_00__T_place_BALCONY`: contribution `-0.007194`
- `lag_05__CT_place_LIBRARY`: contribution `-0.006820`
- `lag_04__T_place_BALCONY`: contribution `-0.005788`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.004591`
- `lag_05__CT3__flash_duration`: contribution `-0.003093`
- `lag_05__CT_flash_duration_sum`: contribution `-0.002407`
- `lag_05__CT2__flash_duration`: contribution `-0.001857`

### tick `136831`, seconds `23.50`, LSTM delta `-0.0794`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `-0.012139`
- `lag_04__CT_place_QUAD`: contribution `-0.010456`
- `lag_01__CT_place_LIBRARY`: contribution `-0.005073`
- `lag_10__CT5__duck_amount`: contribution `-0.003757`
- `lag_04__CT_place_TOPOFMID`: contribution `-0.002480`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `-0.001206`
- `lag_05__CT1__molly`: contribution `-0.001188`

### tick `137503`, seconds `34.00`, LSTM delta `+0.0719`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.011554`
- `lag_00__CT2__shots_fired`: contribution `+0.005915`
- `lag_14__CT3__duck_amount`: contribution `+0.003615`
- `lag_00__CT2__flash_duration`: contribution `+0.003499`
- `lag_04__CT_place_LIBRARY`: contribution `+0.003110`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.003499`
- `lag_04__CT3__flash_duration`: contribution `+0.002014`

### tick `136607`, seconds `20.00`, LSTM delta `+0.0526`

Top all feature movements:
- `lag_11__CT_place_TOPOFMID`: contribution `+0.004246`
- `lag_03__CT5__duck_amount`: contribution `+0.002785`
- `lag_02__CT5__duck_amount`: contribution `+0.002717`
- `lag_12__T5__is_walking`: contribution `+0.002626`
- `lag_08__T1__duck_amount`: contribution `+0.002535`

Top utility-only movements:
- No utility movement among the top local contributors.
