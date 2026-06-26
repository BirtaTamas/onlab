# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `130882`, seconds `18.50`, LSTM `0.7858`, delta `+0.1053`
- tick `131010`, seconds `20.50`, LSTM `0.8758`, delta `+0.0723`
- tick `135042`, seconds `83.50`, LSTM `0.9675`, delta `+0.0416`
- tick `133954`, seconds `66.50`, LSTM `0.9548`, delta `+0.0382`
- tick `131202`, seconds `23.50`, LSTM `0.8511`, delta `-0.0314`
- tick `131874`, seconds `34.00`, LSTM `0.8779`, delta `+0.0284`
- tick `132642`, seconds `46.00`, LSTM `0.9024`, delta `+0.0276`
- tick `132002`, seconds `36.00`, LSTM `0.8903`, delta `+0.0260`
- tick `131714`, seconds `31.50`, LSTM `0.8471`, delta `+0.0246`
- tick `131682`, seconds `31.00`, LSTM `0.8226`, delta `-0.0244`

## Top 15 local ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.001173`, |coef| `0.001173`
- `lag_13__CT_place_UNDERPASS`: coefficient `0.001007`, |coef| `0.001007`
- `lag_00__CT_kills_last_3s`: coefficient `0.000978`, |coef| `0.000978`
- `lag_00__damage_diff_last_5s`: coefficient `0.000827`, |coef| `0.000827`
- `lag_03__CT_place_JUNGLE`: coefficient `0.000817`, |coef| `0.000817`
- `lag_00__kill_diff_last_3s`: coefficient `0.000815`, |coef| `0.000815`
- `lag_11__CT_place_PALACEALLEY`: coefficient `0.000766`, |coef| `0.000766`
- `lag_00__CT_damage_last_5s`: coefficient `0.000760`, |coef| `0.000760`
- `lag_01__CT5__is_scoped`: coefficient `0.000759`, |coef| `0.000759`
- `lag_06__CT_place_JUNGLE`: coefficient `0.000752`, |coef| `0.000752`
- `lag_13__CT_place_TRUCK`: coefficient `0.000717`, |coef| `0.000717`
- `lag_06__CT_place_SNIPERSNEST`: coefficient `-0.000683`, |coef| `0.000683`
- `lag_04__T5__is_walking`: coefficient `0.000663`, |coef| `0.000663`
- `lag_14__CT_place_UNDERPASS`: coefficient `0.000655`, |coef| `0.000655`
- `lag_00__CT2__duck_amount`: coefficient `0.000642`, |coef| `0.000642`

## Top 10 utility ridge features

- `lag_15__T1__flash_duration`: coefficient `-0.000519` (lowers CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `-0.000423` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `-0.000402` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000386` (lowers CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.000378` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `-0.000375` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000354` (raises CT win probability)
- `lag_08__CT3__molly`: coefficient `-0.000352` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000341` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.000331` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.001173` (raises CT win probability)
- `lag_13__CT_place_UNDERPASS`: coefficient `0.001007` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000978` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000827` (raises CT win probability)
- `lag_03__CT_place_JUNGLE`: coefficient `0.000817` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000815` (raises CT win probability)
- `lag_11__CT_place_PALACEALLEY`: coefficient `0.000766` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000760` (raises CT win probability)
- `lag_01__CT5__is_scoped`: coefficient `0.000759` (raises CT win probability)
- `lag_06__CT_place_JUNGLE`: coefficient `0.000752` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `130882`, seconds `18.50`, LSTM delta `+0.1053`

Top all feature movements:
- `lag_13__CT_place_UNDERPASS`: contribution `+0.005840`
- `lag_03__CT_place_JUNGLE`: contribution `+0.005240`
- `lag_13__CT_place_TRUCK`: contribution `+0.004622`
- `lag_06__CT_place_SNIPERSNEST`: contribution `+0.003659`
- `lag_00__CT_kills_last_3s`: contribution `+0.002825`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.002443`

### tick `131010`, seconds `20.50`, LSTM delta `+0.0723`

Top all feature movements:
- `lag_07__CT_place_JUNGLE`: contribution `+0.003150`
- `lag_00__CT_kills_last_3s`: contribution `+0.002825`
- `lag_01__CT5__is_scoped`: contribution `+0.002714`
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.002603`
- `lag_00__kill_diff_last_3s`: contribution `+0.001962`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001095`

### tick `135042`, seconds `83.50`, LSTM delta `+0.0416`

Top all feature movements:
- `lag_11__CT_place_PALACEALLEY`: contribution `+0.011696`
- `lag_00__CT_kills_last_3s`: contribution `+0.002825`
- `lag_00__CT5__flash_duration`: contribution `+0.002108`
- `lag_06__CT5__is_scoped`: contribution `+0.002049`
- `lag_00__kill_diff_last_3s`: contribution `+0.001962`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.002108`
- `lag_11__CT5__flash_duration`: contribution `+0.001831`

### tick `133954`, seconds `66.50`, LSTM delta `+0.0382`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.002825`
- `lag_03__T_place_CONNECTOR`: contribution `+0.002337`
- `lag_06__CT5__is_scoped`: contribution `+0.002049`
- `lag_00__kill_diff_last_3s`: contribution `+0.001962`
- `lag_11__CT_place_JUNGLE`: contribution `+0.001804`

Top utility-only movements:
- `lag_14__T5__flash_duration`: contribution `+0.001725`

### tick `131202`, seconds `23.50`, LSTM delta `-0.0314`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.007567`
- `lag_13__CT_place_UNDERPASS`: contribution `-0.005840`
- `lag_00__CT_kills_last_3s`: contribution `-0.002825`
- `lag_06__CT5__is_scoped`: contribution `-0.002049`
- `lag_13__CT_place_CATWALK`: contribution `-0.001970`

Top utility-only movements:
- No utility movement among the top local contributors.
