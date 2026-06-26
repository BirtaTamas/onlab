# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `8`

## Largest probability jumps

- tick `54400`, seconds `48.50`, LSTM `0.8363`, delta `+0.1478`
- tick `54432`, seconds `49.00`, LSTM `0.9478`, delta `+0.1115`
- tick `52672`, seconds `21.50`, LSTM `0.5285`, delta `+0.0956`
- tick `54368`, seconds `48.00`, LSTM `0.6885`, delta `+0.0705`
- tick `52864`, seconds `24.50`, LSTM `0.5122`, delta `-0.0677`
- tick `52256`, seconds `15.00`, LSTM `0.3834`, delta `-0.0521`
- tick `52416`, seconds `17.50`, LSTM `0.3486`, delta `-0.0399`
- tick `52832`, seconds `24.00`, LSTM `0.5799`, delta `+0.0395`
- tick `52288`, seconds `15.50`, LSTM `0.4222`, delta `+0.0389`
- tick `52640`, seconds `21.00`, LSTM `0.4329`, delta `+0.0377`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002276`, |coef| `0.002276`
- `lag_08__CT_place_SIDEENTRANCE`: coefficient `0.002073`, |coef| `0.002073`
- `lag_00__kill_diff_last_3s`: coefficient `0.002000`, |coef| `0.002000`
- `lag_00__damage_diff_last_5s`: coefficient `0.001801`, |coef| `0.001801`
- `lag_08__T3__duck_amount`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_00__CT_damage_last_5s`: coefficient `0.001594`, |coef| `0.001594`
- `lag_00__CT1__shots_fired`: coefficient `0.001590`, |coef| `0.001590`
- `lag_01__CT4__is_scoped`: coefficient `0.001496`, |coef| `0.001496`
- `lag_04__T_place_TSIDELOWER`: coefficient `-0.001463`, |coef| `0.001463`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001414`, |coef| `0.001414`
- `lag_00__T_place_TSIDEUPPER`: coefficient `-0.001392`, |coef| `0.001392`
- `lag_07__CT_place_SIDEENTRANCE`: coefficient `0.001358`, |coef| `0.001358`
- `lag_00__T2__alive`: coefficient `-0.001307`, |coef| `0.001307`
- `lag_15__CT5__duck_amount`: coefficient `-0.001302`, |coef| `0.001302`
- `lag_09__CT_place_SIDEENTRANCE`: coefficient `0.001288`, |coef| `0.001288`

## Top 10 utility ridge features

- `lag_00__T2__smoke`: coefficient `-0.001249` (lowers CT win probability)
- `lag_12__CT5__smoke`: coefficient `-0.001047` (lowers CT win probability)
- `lag_08__CT_B_site_active_smokes`: coefficient `0.001046` (raises CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `-0.000941` (lowers CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `-0.000869` (lowers CT win probability)
- `lag_11__CT5__smoke`: coefficient `-0.000781` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000752` (lowers CT win probability)
- `lag_07__CT_B_site_active_smokes`: coefficient `0.000749` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `-0.000711` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.000685` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002276` (raises CT win probability)
- `lag_08__CT_place_SIDEENTRANCE`: coefficient `0.002073` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002000` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001801` (raises CT win probability)
- `lag_08__T3__duck_amount`: coefficient `-0.001688` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001594` (raises CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `0.001590` (raises CT win probability)
- `lag_01__CT4__is_scoped`: coefficient `0.001496` (raises CT win probability)
- `lag_04__T_place_TSIDELOWER`: coefficient `-0.001463` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001414` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `54400`, seconds `48.50`, LSTM delta `+0.1478`

Top all feature movements:
- `lag_08__CT_place_SIDEENTRANCE`: contribution `+0.008343`
- `lag_00__CT_kills_last_3s`: contribution `+0.006571`
- `lag_08__T3__duck_amount`: contribution `+0.006365`
- `lag_04__T_place_TSIDELOWER`: contribution `+0.005485`
- `lag_01__CT4__is_scoped`: contribution `+0.005100`

Top utility-only movements:
- `lag_00__T2__smoke`: contribution `+0.002743`
- `lag_12__CT5__smoke`: contribution `+0.002298`

### tick `54432`, seconds `49.00`, LSTM delta `+0.1115`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006571`
- `lag_09__CT_place_SIDEENTRANCE`: contribution `+0.005183`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004911`
- `lag_00__kill_diff_last_3s`: contribution `+0.004813`
- `lag_00__CT1__shots_fired`: contribution `+0.004200`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52672`, seconds `21.50`, LSTM delta `+0.0956`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006571`
- `lag_14__CT_shots_fired_sum`: contribution `+0.005988`
- `lag_01__CT1__flash_duration`: contribution `+0.005121`
- `lag_01__CT4__is_scoped`: contribution `+0.005100`
- `lag_00__kill_diff_last_3s`: contribution `+0.004813`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.005121`
- `lag_12__T3__flash_duration`: contribution `+0.004777`
- `lag_14__CT1__flash_duration`: contribution `+0.002466`
- `lag_14__CT2__flash_duration`: contribution `+0.002160`
- `lag_03__CT3__flash_duration`: contribution `+0.001511`

### tick `54368`, seconds `48.00`, LSTM delta `+0.0705`

Top all feature movements:
- `lag_07__CT_place_SIDEENTRANCE`: contribution `+0.005465`
- `lag_07__T3__duck_amount`: contribution `+0.004475`
- `lag_03__T_place_TSIDELOWER`: contribution `+0.003924`
- `lag_14__CT5__duck_amount`: contribution `+0.003788`
- `lag_00__damage_diff_last_5s`: contribution `+0.003656`

Top utility-only movements:
- `lag_11__CT5__smoke`: contribution `+0.001713`

### tick `52864`, seconds `24.50`, LSTM delta `-0.0677`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009626`
- `lag_00__CT_kills_last_3s`: contribution `-0.006571`
- `lag_04__T_place_TSIDELOWER`: contribution `-0.005485`
- `lag_01__CT4__is_scoped`: contribution `-0.005100`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004911`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `-0.002369`
- `lag_07__CT1__flash_duration`: contribution `-0.002211`
- `lag_00__CT3__flash_duration`: contribution `-0.002065`
