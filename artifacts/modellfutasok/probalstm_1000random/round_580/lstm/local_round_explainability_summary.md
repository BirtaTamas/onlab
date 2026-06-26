# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m3-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `50927`, seconds `66.00`, LSTM `0.7455`, delta `+0.4049`
- tick `52367`, seconds `88.50`, LSTM `0.7938`, delta `+0.3535`
- tick `49199`, seconds `39.00`, LSTM `0.8479`, delta `+0.2562`
- tick `50799`, seconds `64.00`, LSTM `0.5171`, delta `-0.2156`
- tick `51055`, seconds `68.00`, LSTM `0.5815`, delta `-0.1849`
- tick `49039`, seconds `36.50`, LSTM `0.5827`, delta `+0.1458`
- tick `52431`, seconds `89.50`, LSTM `0.9086`, delta `+0.0980`
- tick `50895`, seconds `65.50`, LSTM `0.3406`, delta `-0.0848`
- tick `51279`, seconds `71.50`, LSTM `0.5898`, delta `-0.0777`
- tick `52239`, seconds `86.50`, LSTM `0.4240`, delta `-0.0679`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005522`, |coef| `0.005522`
- `lag_00__T_place_HUT`: coefficient `-0.004838`, |coef| `0.004838`
- `lag_00__CT_kills_last_3s`: coefficient `0.004398`, |coef| `0.004398`
- `lag_00__CT_place_HEAVEN`: coefficient `0.004187`, |coef| `0.004187`
- `lag_00__damage_diff_last_5s`: coefficient `0.003945`, |coef| `0.003945`
- `lag_00__CT_place_VENTS`: coefficient `-0.003344`, |coef| `0.003344`
- `lag_04__T_bomb_zone_count`: coefficient `-0.003128`, |coef| `0.003128`
- `lag_11__T_velocity_mean`: coefficient `-0.003121`, |coef| `0.003121`
- `lag_14__CT_place_LOCKERROOM`: coefficient `0.002980`, |coef| `0.002980`
- `lag_04__CT_place_HEAVEN`: coefficient `-0.002900`, |coef| `0.002900`
- `lag_05__T_place_HUT`: coefficient `-0.002743`, |coef| `0.002743`
- `lag_02__CT_place_HEAVEN`: coefficient `-0.002711`, |coef| `0.002711`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002701`, |coef| `0.002701`
- `lag_14__T_place_TROPHY`: coefficient `0.002697`, |coef| `0.002697`
- `lag_15__CT5__is_walking`: coefficient `-0.002528`, |coef| `0.002528`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001745` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001522` (lowers CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.001430` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.001145` (lowers CT win probability)
- `lag_07__CT2__flash`: coefficient `-0.001047` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000940` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000882` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.000869` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.000854` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.000816` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005522` (raises CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.004838` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004398` (raises CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `0.004187` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003945` (raises CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `-0.003344` (lowers CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `-0.003128` (lowers CT win probability)
- `lag_11__T_velocity_mean`: coefficient `-0.003121` (lowers CT win probability)
- `lag_14__CT_place_LOCKERROOM`: coefficient `0.002980` (raises CT win probability)
- `lag_04__CT_place_HEAVEN`: coefficient `-0.002900` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `50927`, seconds `66.00`, LSTM delta `+0.4049`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `+0.045096`
- `lag_00__CT_place_HEAVEN`: contribution `+0.022607`
- `lag_01__T_place_HUT`: contribution `+0.020133`
- `lag_03__T_place_HUT`: contribution `+0.019635`
- `lag_02__CT_place_HEAVEN`: contribution `+0.014636`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52367`, seconds `88.50`, LSTM delta `+0.3535`

Top all feature movements:
- `lag_00__CT_place_VENTS`: contribution `+0.028062`
- `lag_00__CT_place_HEAVEN`: contribution `+0.022607`
- `lag_11__T_velocity_mean`: contribution `+0.021109`
- `lag_04__T_bomb_zone_count`: contribution `+0.018210`
- `lag_04__CT_place_HEAVEN`: contribution `+0.015660`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `49199`, seconds `39.00`, LSTM delta `+0.2562`

Top all feature movements:
- `lag_05__CT_place_CONTROL`: contribution `+0.021456`
- `lag_00__kill_diff_last_3s`: contribution `+0.013291`
- `lag_00__CT_kills_last_3s`: contribution `+0.012697`
- `lag_00__T_place_TROPHY`: contribution `+0.009602`
- `lag_08__CT_place_MINI`: contribution `+0.009148`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50799`, seconds `64.00`, LSTM delta `-0.2156`

Top all feature movements:
- `lag_14__CT_place_LOCKERROOM`: contribution `-0.037097`
- `lag_00__kill_diff_last_3s`: contribution `-0.013291`
- `lag_01__CT_place_HELL`: contribution `-0.011499`
- `lag_00__damage_diff_last_5s`: contribution `-0.008900`
- `lag_00__T_kills_last_3s`: contribution `-0.007738`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51055`, seconds `68.00`, LSTM delta `-0.1849`

Top all feature movements:
- `lag_05__T_place_HUT`: contribution `-0.025572`
- `lag_04__CT_place_HEAVEN`: contribution `-0.015660`
- `lag_00__kill_diff_last_3s`: contribution `-0.013291`
- `lag_00__CT_place_RAFTERS`: contribution `+0.011369`
- `lag_04__T_place_HUT`: contribution `-0.010514`

Top utility-only movements:
- No utility movement among the top local contributors.
