# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `57342`, seconds `37.50`, LSTM `0.0858`, delta `-0.2563`
- tick `58174`, seconds `50.50`, LSTM `0.1005`, delta `-0.2148`
- tick `57790`, seconds `44.50`, LSTM `0.2556`, delta `+0.1442`
- tick `56702`, seconds `27.50`, LSTM `0.3901`, delta `-0.1376`
- tick `58142`, seconds `50.00`, LSTM `0.3152`, delta `+0.1288`
- tick `57278`, seconds `36.50`, LSTM `0.3257`, delta `+0.1035`
- tick `56766`, seconds `28.50`, LSTM `0.2635`, delta `-0.0738`
- tick `57982`, seconds `47.50`, LSTM `0.2457`, delta `-0.0668`
- tick `56734`, seconds `28.00`, LSTM `0.3373`, delta `-0.0528`
- tick `57854`, seconds `45.50`, LSTM `0.3223`, delta `+0.0467`

## Top 15 local ridge features

- `lag_02__CT_place_SCAFFOLDING`: coefficient `0.002677`, |coef| `0.002677`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002171`, |coef| `0.002171`
- `lag_10__CT_place_TRUCK`: coefficient `-0.002113`, |coef| `0.002113`
- `lag_00__damage_diff_last_5s`: coefficient `0.001953`, |coef| `0.001953`
- `lag_00__kill_diff_last_3s`: coefficient `0.001864`, |coef| `0.001864`
- `lag_00__T_kills_last_3s`: coefficient `-0.001819`, |coef| `0.001819`
- `lag_00__T_damage_last_5s`: coefficient `-0.001721`, |coef| `0.001721`
- `lag_07__CT_place_STAIRS`: coefficient `0.001698`, |coef| `0.001698`
- `lag_00__CT_place_UNDERPASS`: coefficient `0.001669`, |coef| `0.001669`
- `lag_13__T3__shots_fired`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_00__T4__is_scoped`: coefficient `0.001271`, |coef| `0.001271`
- `lag_00__T3__duck_amount`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_14__CT1__flash_duration`: coefficient `0.001258`, |coef| `0.001258`
- `lag_02__CT_place_UNDERPASS`: coefficient `0.001237`, |coef| `0.001237`
- `lag_00__T4__duck_amount`: coefficient `0.001228`, |coef| `0.001228`

## Top 10 utility ridge features

- `lag_14__CT1__flash_duration`: coefficient `0.001258` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001091` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.001050` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001030` (raises CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.001004` (raises CT win probability)
- `lag_00__CT5__molly`: coefficient `0.000955` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000901` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `-0.000897` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000890` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.000847` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_SCAFFOLDING`: coefficient `0.002677` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002171` (lowers CT win probability)
- `lag_10__CT_place_TRUCK`: coefficient `-0.002113` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001953` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001864` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001819` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001721` (lowers CT win probability)
- `lag_07__CT_place_STAIRS`: coefficient `0.001698` (raises CT win probability)
- `lag_00__CT_place_UNDERPASS`: coefficient `0.001669` (raises CT win probability)
- `lag_13__T3__shots_fired`: coefficient `-0.001486` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `57342`, seconds `37.50`, LSTM delta `-0.2563`

Top all feature movements:
- `lag_02__CT_place_SCAFFOLDING`: contribution `-0.055870`
- `lag_00__T_shots_fired_sum`: contribution `-0.017904`
- `lag_02__CT4__flash_duration`: contribution `-0.008049`
- `lag_08__CT_place_JUNGLE`: contribution `-0.007389`
- `lag_00__T_kills_last_3s`: contribution `-0.005764`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.008049`
- `lag_00__CT2__flash`: contribution `-0.003725`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.003449`

### tick `58174`, seconds `50.50`, LSTM delta `-0.2148`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `-0.013214`
- `lag_02__CT_place_JUNGLE`: contribution `-0.007130`
- `lag_14__CT1__flash_duration`: contribution `-0.006250`
- `lag_00__T4__is_scoped`: contribution `-0.005905`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.005797`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.006250`

### tick `57790`, seconds `44.50`, LSTM delta `+0.1442`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.008138`
- `lag_12__T1__shots_fired`: contribution `+0.007639`
- `lag_09__CT_place_JUNGLE`: contribution `+0.007270`
- `lag_13__T3__shots_fired`: contribution `+0.006298`
- `lag_00__T3__duck_amount`: contribution `+0.004748`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.002891`
- `lag_10__T5__flash_duration`: contribution `+0.002684`
- `lag_10__CT2__flash_duration`: contribution `+0.002684`
- `lag_05__CT4__flash_duration`: contribution `+0.002623`
- `lag_10__CT1__flash_duration`: contribution `+0.002528`

### tick `56702`, seconds `27.50`, LSTM delta `-0.1376`

Top all feature movements:
- `lag_10__CT_place_TRUCK`: contribution `-0.013632`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.009677`
- `lag_00__T4__is_scoped`: contribution `-0.005905`
- `lag_00__T_kills_last_3s`: contribution `-0.005764`
- `lag_14__CT_place_TRUCK`: contribution `-0.005240`

Top utility-only movements:
- `lag_00__CT5__utility_total`: contribution `-0.003092`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.003080`
- `lag_00__CT5__molly`: contribution `-0.002369`

### tick `58142`, seconds `50.00`, LSTM delta `+0.1288`

Top all feature movements:
- `lag_05__CT_flashed_players`: contribution `+0.005032`
- `lag_00__T4__duck_amount`: contribution `+0.004541`
- `lag_12__T3__duck_amount`: contribution `+0.004350`
- `lag_08__T1__duck_amount`: contribution `+0.004137`
- `lag_01__T4__is_scoped`: contribution `+0.003991`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `+0.002599`
