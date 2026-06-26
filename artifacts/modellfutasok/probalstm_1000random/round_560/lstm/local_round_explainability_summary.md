# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `52218`, seconds `54.00`, LSTM `0.1115`, delta `-0.3094`
- tick `51354`, seconds `40.50`, LSTM `0.1678`, delta `-0.2627`
- tick `51706`, seconds `46.00`, LSTM `0.3561`, delta `+0.1697`
- tick `49626`, seconds `13.50`, LSTM `0.5309`, delta `-0.0977`
- tick `51738`, seconds `46.50`, LSTM `0.4522`, delta `+0.0961`
- tick `50938`, seconds `34.00`, LSTM `0.4654`, delta `+0.0797`
- tick `50394`, seconds `25.50`, LSTM `0.3148`, delta `-0.0700`
- tick `50330`, seconds `24.50`, LSTM `0.3845`, delta `-0.0665`
- tick `52282`, seconds `55.00`, LSTM `0.0120`, delta `-0.0562`
- tick `50234`, seconds `23.00`, LSTM `0.4374`, delta `-0.0546`

## Top 15 local ridge features

- `lag_12__CT_place_QUAD`: coefficient `0.002984`, |coef| `0.002984`
- `lag_07__T1__is_scoped`: coefficient `-0.002456`, |coef| `0.002456`
- `lag_00__T_kills_last_3s`: coefficient `-0.002381`, |coef| `0.002381`
- `lag_04__CT_place_QUAD`: coefficient `0.002333`, |coef| `0.002333`
- `lag_00__T2__flash_duration`: coefficient `0.002285`, |coef| `0.002285`
- `lag_05__T4__duck_amount`: coefficient `0.002262`, |coef| `0.002262`
- `lag_00__kill_diff_last_3s`: coefficient `0.002253`, |coef| `0.002253`
- `lag_00__T4__flash_duration`: coefficient `0.002239`, |coef| `0.002239`
- `lag_00__damage_diff_last_5s`: coefficient `0.002203`, |coef| `0.002203`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002164`, |coef| `0.002164`
- `lag_01__T1__is_scoped`: coefficient `0.002113`, |coef| `0.002113`
- `lag_07__CT3__is_scoped`: coefficient `-0.001975`, |coef| `0.001975`
- `lag_00__T_flash_duration_sum`: coefficient `0.001904`, |coef| `0.001904`
- `lag_00__CT1__is_walking`: coefficient `0.001851`, |coef| `0.001851`
- `lag_06__T4__duck_amount`: coefficient `-0.001769`, |coef| `0.001769`

## Top 10 utility ridge features

- `lag_00__T2__flash_duration`: coefficient `0.002285` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.002239` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.001904` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `-0.001534` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `-0.001515` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.001486` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `-0.001326` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001200` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001159` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `-0.001045` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_QUAD`: coefficient `0.002984` (raises CT win probability)
- `lag_07__T1__is_scoped`: coefficient `-0.002456` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002381` (lowers CT win probability)
- `lag_04__CT_place_QUAD`: coefficient `0.002333` (raises CT win probability)
- `lag_05__T4__duck_amount`: coefficient `0.002262` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002253` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002203` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002164` (raises CT win probability)
- `lag_01__T1__is_scoped`: coefficient `0.002113` (raises CT win probability)
- `lag_07__CT3__is_scoped`: coefficient `-0.001975` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `52218`, seconds `54.00`, LSTM delta `-0.3094`

Top all feature movements:
- `lag_12__CT_place_QUAD`: contribution `-0.023516`
- `lag_07__T1__is_scoped`: contribution `-0.014031`
- `lag_01__T1__is_scoped`: contribution `-0.012071`
- `lag_00__CT_shots_fired_sum`: contribution `-0.012028`
- `lag_07__CT3__is_scoped`: contribution `-0.008982`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51354`, seconds `40.50`, LSTM delta `-0.2627`

Top all feature movements:
- `lag_00__T2__flash_duration`: contribution `-0.016991`
- `lag_00__T4__flash_duration`: contribution `-0.016601`
- `lag_00__T_flash_duration_sum`: contribution `-0.011537`
- `lag_13__T2__flash_duration`: contribution `-0.011408`
- `lag_13__T4__flash_duration`: contribution `-0.011017`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `-0.016991`
- `lag_00__T4__flash_duration`: contribution `-0.016601`
- `lag_00__T_flash_duration_sum`: contribution `-0.011537`
- `lag_13__T2__flash_duration`: contribution `-0.011408`
- `lag_13__T4__flash_duration`: contribution `-0.011017`

### tick `51706`, seconds `46.00`, LSTM delta `+0.1697`

Top all feature movements:
- `lag_04__CT_place_QUAD`: contribution `+0.018384`
- `lag_11__T1__is_scoped`: contribution `+0.009979`
- `lag_11__T2__flash_duration`: contribution `+0.007768`
- `lag_11__T4__flash_duration`: contribution `+0.007418`
- `lag_00__kill_diff_last_3s`: contribution `+0.005424`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.007768`
- `lag_11__T4__flash_duration`: contribution `+0.007418`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.005206`
- `lag_11__T_flash_duration_sum`: contribution `+0.004188`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.002625`

### tick `49626`, seconds `13.50`, LSTM delta `-0.0977`

Top all feature movements:
- `lag_14__CT_smokes_last_5s`: contribution `-0.013857`
- `lag_00__T_kills_last_3s`: contribution `-0.007543`
- `lag_13__T_place_LOWERMID`: contribution `-0.007288`
- `lag_00__T_flashed_players`: contribution `-0.005821`
- `lag_13__T_place_SECONDMID`: contribution `+0.005645`

Top utility-only movements:
- `lag_14__CT_smokes_last_5s`: contribution `-0.013857`
- `lag_00__T2__flash_duration`: contribution `-0.002164`

### tick `51738`, seconds `46.50`, LSTM delta `+0.0961`

Top all feature movements:
- `lag_05__CT_place_QUAD`: contribution `+0.011640`
- `lag_00__T_place_SECONDMID`: contribution `+0.005726`
- `lag_12__T1__is_scoped`: contribution `+0.005289`
- `lag_08__CT_place_TOPOFMID`: contribution `+0.004969`
- `lag_12__T4__flash_duration`: contribution `+0.004415`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `+0.004415`
- `lag_12__T2__flash_duration`: contribution `+0.004373`
- `lag_12__T_flash_duration_sum`: contribution `+0.002479`
- `lag_01__CT_B_site_active_infernos`: contribution `+0.002338`
