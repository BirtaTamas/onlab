# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `70596`, seconds `28.00`, LSTM `0.0702`, delta `-0.3484`
- tick `72260`, seconds `54.00`, LSTM `0.8076`, delta `+0.3263`
- tick `70116`, seconds `20.50`, LSTM `0.4041`, delta `-0.2987`
- tick `71844`, seconds `47.50`, LSTM `0.3761`, delta `+0.2415`
- tick `70308`, seconds `23.50`, LSTM `0.4765`, delta `+0.0911`
- tick `72388`, seconds `56.00`, LSTM `0.9243`, delta `+0.0766`
- tick `70148`, seconds `21.00`, LSTM `0.4781`, delta `+0.0740`
- tick `71204`, seconds `37.50`, LSTM `0.0782`, delta `+0.0654`
- tick `70564`, seconds `27.50`, LSTM `0.4186`, delta `-0.0598`
- tick `70244`, seconds `22.50`, LSTM `0.4189`, delta `-0.0428`

## Top 15 local ridge features

- `lag_02__T_place_PIT`: coefficient `-0.003405`, |coef| `0.003405`
- `lag_11__CT_place_QUAD`: coefficient `-0.003141`, |coef| `0.003141`
- `lag_01__T_place_BALCONY`: coefficient `-0.003064`, |coef| `0.003064`
- `lag_00__T_place_GRAVEYARD`: coefficient `-0.002850`, |coef| `0.002850`
- `lag_11__T_place_GRAVEYARD`: coefficient `0.002706`, |coef| `0.002706`
- `lag_15__T_place_PIT`: coefficient `-0.002697`, |coef| `0.002697`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002626`, |coef| `0.002626`
- `lag_02__T_place_BALCONY`: coefficient `-0.002452`, |coef| `0.002452`
- `lag_00__kill_diff_last_3s`: coefficient `0.002404`, |coef| `0.002404`
- `lag_00__damage_diff_last_5s`: coefficient `0.002261`, |coef| `0.002261`
- `lag_03__CT_place_ARCH`: coefficient `-0.002196`, |coef| `0.002196`
- `lag_00__T_kills_last_3s`: coefficient `-0.002042`, |coef| `0.002042`
- `lag_03__T_place_ARCH`: coefficient `-0.001948`, |coef| `0.001948`
- `lag_04__T_place_PIT`: coefficient `-0.001935`, |coef| `0.001935`
- `lag_03__T_place_PIT`: coefficient `-0.001863`, |coef| `0.001863`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002626` (lowers CT win probability)
- `lag_08__CT1__molly`: coefficient `-0.001384` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001253` (lowers CT win probability)
- `lag_14__CT1__smoke`: coefficient `-0.001186` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001167` (raises CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `-0.001159` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `0.001143` (raises CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.001111` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.001091` (raises CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001039` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_PIT`: coefficient `-0.003405` (lowers CT win probability)
- `lag_11__CT_place_QUAD`: coefficient `-0.003141` (lowers CT win probability)
- `lag_01__T_place_BALCONY`: coefficient `-0.003064` (lowers CT win probability)
- `lag_00__T_place_GRAVEYARD`: coefficient `-0.002850` (lowers CT win probability)
- `lag_11__T_place_GRAVEYARD`: coefficient `0.002706` (raises CT win probability)
- `lag_15__T_place_PIT`: coefficient `-0.002697` (lowers CT win probability)
- `lag_02__T_place_BALCONY`: coefficient `-0.002452` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002404` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002261` (raises CT win probability)
- `lag_03__CT_place_ARCH`: coefficient `-0.002196` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `70596`, seconds `28.00`, LSTM delta `-0.3484`

Top all feature movements:
- `lag_01__T_place_BALCONY`: contribution `-0.042132`
- `lag_02__T_place_BALCONY`: contribution `-0.033713`
- `lag_10__T_place_ARCH`: contribution `-0.011872`
- `lag_15__T_place_ARCH`: contribution `-0.009167`
- `lag_15__CT3__flash_duration`: contribution `-0.009129`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `-0.009129`
- `lag_05__CT5__flash_duration`: contribution `-0.005889`
- `lag_14__CT5__flash_duration`: contribution `-0.005852`
- `lag_15__T4__flash_duration`: contribution `-0.004828`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.004625`

### tick `72260`, seconds `54.00`, LSTM delta `+0.3263`

Top all feature movements:
- `lag_00__T_place_GRAVEYARD`: contribution `+0.056024`
- `lag_11__T_place_GRAVEYARD`: contribution `+0.053194`
- `lag_15__T_place_PIT`: contribution `+0.017018`
- `lag_00__T_flash_alpha_mean`: contribution `+0.015930`
- `lag_13__T_duck_amount_mean`: contribution `+0.008096`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.015930`

### tick `70116`, seconds `20.50`, LSTM delta `-0.2987`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `-0.018127`
- `lag_00__T_kills_last_3s`: contribution `-0.012939`
- `lag_00__T_place_ARCH`: contribution `-0.009834`
- `lag_10__CT3__flash_duration`: contribution `-0.008866`
- `lag_00__CT3__flash_duration`: contribution `-0.006478`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.008866`
- `lag_00__CT3__flash_duration`: contribution `-0.006478`
- `lag_10__T4__flash_duration`: contribution `-0.004661`
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.003728`
- `lag_00__T4__flash_duration`: contribution `-0.003414`

### tick `71844`, seconds `47.50`, LSTM delta `+0.2415`

Top all feature movements:
- `lag_11__CT_place_QUAD`: contribution `+0.024759`
- `lag_02__T_place_PIT`: contribution `+0.021489`
- `lag_03__CT_place_ARCH`: contribution `+0.008962`
- `lag_00__T_duck_amount_mean`: contribution `+0.008110`
- `lag_15__T4__is_scoped`: contribution `+0.006791`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `+0.003851`
- `lag_08__CT1__molly`: contribution `+0.003445`

### tick `70308`, seconds `23.50`, LSTM delta `+0.0911`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `+0.012939`
- `lag_06__T_place_ARCH`: contribution `+0.009662`
- `lag_05__CT5__flash_duration`: contribution `+0.005889`
- `lag_00__kill_diff_last_3s`: contribution `+0.005785`
- `lag_06__CT_place_ARCH`: contribution `+0.004105`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `+0.005889`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.003851`
- `lag_06__CT3__flash_duration`: contribution `+0.002406`
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.002175`
- `lag_08__CT_A_site_active_infernos`: contribution `+0.001710`
