# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `14236`, seconds `85.00`, LSTM `0.8576`, delta `+0.2082`
- tick `9468`, seconds `10.50`, LSTM `0.7060`, delta `+0.0969`
- tick `9436`, seconds `10.00`, LSTM `0.6091`, delta `+0.0951`
- tick `8892`, seconds `1.50`, LSTM `0.6194`, delta `+0.0689`
- tick `9564`, seconds `12.00`, LSTM `0.7191`, delta `+0.0600`
- tick `10524`, seconds `27.00`, LSTM `0.5826`, delta `-0.0578`
- tick `14748`, seconds `93.00`, LSTM `0.9664`, delta `+0.0468`
- tick `14364`, seconds `87.00`, LSTM `0.9700`, delta `+0.0450`
- tick `10332`, seconds `24.00`, LSTM `0.6917`, delta `-0.0405`
- tick `14684`, seconds `92.00`, LSTM `0.9223`, delta `-0.0384`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002916`, |coef| `0.002916`
- `lag_12__T_place_CATWALK`: coefficient `0.002879`, |coef| `0.002879`
- `lag_03__T_B_site_active_infernos`: coefficient `0.002778`, |coef| `0.002778`
- `lag_08__T_A_site_active_infernos`: coefficient `-0.002738`, |coef| `0.002738`
- `lag_00__T_place_CATWALK`: coefficient `-0.002680`, |coef| `0.002680`
- `lag_00__T4__alive`: coefficient `-0.002616`, |coef| `0.002616`
- `lag_00__kill_diff_last_3s`: coefficient `0.002577`, |coef| `0.002577`
- `lag_00__T4__armor`: coefficient `-0.002390`, |coef| `0.002390`
- `lag_00__T4__hp`: coefficient `-0.002308`, |coef| `0.002308`
- `lag_00__damage_diff_last_5s`: coefficient `0.002305`, |coef| `0.002305`
- `lag_00__CT_damage_last_5s`: coefficient `0.002264`, |coef| `0.002264`
- `lag_06__T2__molly`: coefficient `-0.002254`, |coef| `0.002254`
- `lag_00__T4__has_helmet`: coefficient `-0.002222`, |coef| `0.002222`
- `lag_02__T2__is_walking`: coefficient `0.002191`, |coef| `0.002191`
- `lag_08__T_active_infernos`: coefficient `-0.001851`, |coef| `0.001851`

## Top 10 utility ridge features

- `lag_03__T_B_site_active_infernos`: coefficient `0.002778` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `-0.002738` (lowers CT win probability)
- `lag_06__T2__molly`: coefficient `-0.002254` (lowers CT win probability)
- `lag_08__T_active_infernos`: coefficient `-0.001851` (lowers CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `-0.001737` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.001685` (lowers CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.001549` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.001387` (lowers CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.001336` (lowers CT win probability)
- `lag_05__T2__molly`: coefficient `-0.001334` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002916` (raises CT win probability)
- `lag_12__T_place_CATWALK`: coefficient `0.002879` (raises CT win probability)
- `lag_00__T_place_CATWALK`: coefficient `-0.002680` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.002616` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002577` (raises CT win probability)
- `lag_00__T4__armor`: coefficient `-0.002390` (lowers CT win probability)
- `lag_00__T4__hp`: coefficient `-0.002308` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002305` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002264` (raises CT win probability)
- `lag_00__T4__has_helmet`: coefficient `-0.002222` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14236`, seconds `85.00`, LSTM delta `+0.2082`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008418`
- `lag_12__T_place_CATWALK`: contribution `+0.008286`
- `lag_08__T_A_site_active_infernos`: contribution `+0.008151`
- `lag_03__T_B_site_active_infernos`: contribution `+0.007853`
- `lag_00__T_place_CATWALK`: contribution `+0.007714`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.008151`
- `lag_03__T_B_site_active_infernos`: contribution `+0.007853`
- `lag_06__T2__molly`: contribution `+0.005022`
- `lag_08__T_active_infernos`: contribution `+0.003856`

### tick `9468`, seconds `10.50`, LSTM delta `+0.0969`

Top all feature movements:
- `lag_03__T_mollies_last_5s`: contribution `+0.018519`
- `lag_13__T_mollies_last_5s`: contribution `+0.015349`
- `lag_06__T_smokes_last_5s`: contribution `+0.009383`
- `lag_10__CT_place_SHOP`: contribution `+0.006889`
- `lag_01__CT_kills_last_3s`: contribution `+0.003993`

Top utility-only movements:
- `lag_03__T_mollies_last_5s`: contribution `+0.018519`
- `lag_13__T_mollies_last_5s`: contribution `+0.015349`
- `lag_06__T_smokes_last_5s`: contribution `+0.009383`
- `lag_02__T_flashes_last_5s`: contribution `+0.003623`
- `lag_12__T_flashes_last_5s`: contribution `+0.002952`

### tick `9436`, seconds `10.00`, LSTM delta `+0.0951`

Top all feature movements:
- `lag_02__T_mollies_last_5s`: contribution `+0.012625`
- `lag_12__T_mollies_last_5s`: contribution `+0.009067`
- `lag_00__CT_kills_last_3s`: contribution `+0.008418`
- `lag_05__T_smokes_last_5s`: contribution `+0.006426`
- `lag_00__kill_diff_last_3s`: contribution `+0.006203`

Top utility-only movements:
- `lag_02__T_mollies_last_5s`: contribution `+0.012625`
- `lag_12__T_mollies_last_5s`: contribution `+0.009067`
- `lag_05__T_smokes_last_5s`: contribution `+0.006426`
- `lag_15__T_smokes_last_5s`: contribution `+0.004583`
- `lag_01__T_flashes_last_5s`: contribution `+0.002509`

### tick `8892`, seconds `1.50`, LSTM delta `+0.0689`

Top all feature movements:
- `lag_02__CT_velocity_mean`: contribution `+0.005441`
- `lag_02__T2__is_walking`: contribution `+0.005032`
- `lag_03__T_place_TSPAWN`: contribution `+0.003418`
- `lag_03__CT_place_CTSPAWN`: contribution `+0.002446`
- `lag_01__bomb_events_last_5s`: contribution `+0.002344`

Top utility-only movements:
- `lag_00__T3__utility_total`: contribution `+0.001330`
- `lag_03__CT1__flash`: contribution `+0.001187`
- `lag_00__T3__smoke`: contribution `+0.001084`
- `lag_00__T3__molly`: contribution `+0.001082`
- `lag_03__T_flash_alpha_mean`: contribution `+0.001059`

### tick `9564`, seconds `12.00`, LSTM delta `+0.0600`

Top all feature movements:
- `lag_02__T_place_HOUSE`: contribution `+0.005904`
- `lag_06__T_mollies_last_5s`: contribution `+0.005841`
- `lag_00__CT_place_TRUCK`: contribution `+0.005840`
- `lag_15__T_flashes_last_5s`: contribution `+0.003974`
- `lag_00__T_place_HOUSE`: contribution `+0.003970`

Top utility-only movements:
- `lag_06__T_mollies_last_5s`: contribution `+0.005841`
- `lag_15__T_flashes_last_5s`: contribution `+0.003974`
- `lag_09__T_smokes_last_5s`: contribution `+0.002962`
- `lag_00__T1__flash_duration`: contribution `+0.002418`
- `lag_00__T5__flash_duration`: contribution `+0.001968`
