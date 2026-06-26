# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `40862`, seconds `40.50`, LSTM `0.8484`, delta `+0.1500`
- tick `40510`, seconds `35.00`, LSTM `0.7182`, delta `+0.1238`
- tick `41086`, seconds `44.00`, LSTM `0.9357`, delta `+0.0846`
- tick `41470`, seconds `50.00`, LSTM `0.9667`, delta `+0.0408`
- tick `38942`, seconds `10.50`, LSTM `0.5430`, delta `-0.0341`
- tick `39006`, seconds `11.50`, LSTM `0.5290`, delta `-0.0294`
- tick `38782`, seconds `8.00`, LSTM `0.5700`, delta `+0.0290`
- tick `39038`, seconds `12.00`, LSTM `0.5558`, delta `+0.0268`
- tick `39742`, seconds `23.00`, LSTM `0.6011`, delta `+0.0231`
- tick `38910`, seconds `10.00`, LSTM `0.5772`, delta `-0.0228`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001257`, |coef| `0.001257`
- `lag_11__CT_place_HUTROOF`: coefficient `-0.001091`, |coef| `0.001091`
- `lag_05__CT_place_MINI`: coefficient `0.001065`, |coef| `0.001065`
- `lag_00__kill_diff_last_3s`: coefficient `0.001048`, |coef| `0.001048`
- `lag_00__T_place_TROPHY`: coefficient `-0.001035`, |coef| `0.001035`
- `lag_00__damage_diff_last_5s`: coefficient `0.000925`, |coef| `0.000925`
- `lag_00__CT_damage_last_5s`: coefficient `0.000894`, |coef| `0.000894`
- `lag_09__T_place_VENDING`: coefficient `0.000876`, |coef| `0.000876`
- `lag_05__CT_place_ADMIN`: coefficient `0.000861`, |coef| `0.000861`
- `lag_04__CT3__is_scoped`: coefficient `0.000807`, |coef| `0.000807`
- `lag_06__CT_place_ADMIN`: coefficient `0.000744`, |coef| `0.000744`
- `lag_10__T4__is_scoped`: coefficient `0.000730`, |coef| `0.000730`
- `lag_08__CT3__is_scoped`: coefficient `0.000683`, |coef| `0.000683`
- `lag_05__T4__is_scoped`: coefficient `-0.000653`, |coef| `0.000653`
- `lag_05__CT_place_HELL`: coefficient `0.000646`, |coef| `0.000646`

## Top 10 utility ridge features

- `lag_00__T1__molly`: coefficient `-0.000548` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000530` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000477` (lowers CT win probability)
- `lag_13__T4__smoke`: coefficient `-0.000469` (lowers CT win probability)
- `lag_14__CT2__smoke`: coefficient `-0.000464` (lowers CT win probability)
- `lag_02__T4__smoke`: coefficient `-0.000444` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000438` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000431` (lowers CT win probability)
- `lag_11__T1__molly`: coefficient `-0.000428` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000422` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001257` (raises CT win probability)
- `lag_11__CT_place_HUTROOF`: coefficient `-0.001091` (lowers CT win probability)
- `lag_05__CT_place_MINI`: coefficient `0.001065` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001048` (raises CT win probability)
- `lag_00__T_place_TROPHY`: coefficient `-0.001035` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000925` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000894` (raises CT win probability)
- `lag_09__T_place_VENDING`: coefficient `0.000876` (raises CT win probability)
- `lag_05__CT_place_ADMIN`: coefficient `0.000861` (raises CT win probability)
- `lag_04__CT3__is_scoped`: coefficient `0.000807` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `40862`, seconds `40.50`, LSTM delta `+0.1500`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `+0.006562`
- `lag_05__CT_place_MINI`: contribution `+0.006532`
- `lag_05__CT_place_ADMIN`: contribution `+0.005979`
- `lag_09__T_place_VENDING`: contribution `+0.004444`
- `lag_10__CT_place_ADMIN`: contribution `+0.003731`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40510`, seconds `35.00`, LSTM delta `+0.1238`

Top all feature movements:
- `lag_11__CT_place_HUTROOF`: contribution `+0.007635`
- `lag_06__CT_place_ADMIN`: contribution `+0.005167`
- `lag_00__CT_kills_last_3s`: contribution `+0.003630`
- `lag_08__CT3__is_scoped`: contribution `+0.003104`
- `lag_05__T4__is_scoped`: contribution `+0.003032`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41086`, seconds `44.00`, LSTM delta `+0.0846`

Top all feature movements:
- `lag_04__CT3__is_scoped`: contribution `+0.003672`
- `lag_00__CT_kills_last_3s`: contribution `+0.003630`
- `lag_12__CT_place_ADMIN`: contribution `+0.003164`
- `lag_02__T4__flash_duration`: contribution `+0.002782`
- `lag_00__kill_diff_last_3s`: contribution `+0.002523`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.002782`
- `lag_02__T3__flash_duration`: contribution `+0.002486`
- `lag_02__T_flash_duration_sum`: contribution `+0.001749`
- `lag_00__T4__flash_duration`: contribution `+0.001593`

### tick `41470`, seconds `50.00`, LSTM delta `+0.0408`

Top all feature movements:
- `lag_06__CT_place_ADMIN`: contribution `-0.005167`
- `lag_04__CT3__is_scoped`: contribution `+0.003672`
- `lag_00__CT_kills_last_3s`: contribution `+0.003630`
- `lag_12__CT_place_ADMIN`: contribution `+0.003164`
- `lag_10__CT_place_MINI`: contribution `+0.003033`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `+0.001243`
- `lag_03__T3__flash_duration`: contribution `-0.000950`

### tick `38942`, seconds `10.50`, LSTM delta `-0.0341`

Top all feature movements:
- `lag_00__CT_flashed_players`: contribution `-0.003705`
- `lag_05__CT_place_HELL`: contribution `-0.003503`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003122`
- `lag_10__CT_place_OUTSIDE`: contribution `-0.002372`
- `lag_08__CT_place_HEAVEN`: contribution `-0.001998`

Top utility-only movements:
- `lag_00__CT_flash_duration_sum`: contribution `-0.003122`
- `lag_00__CT2__flash_duration`: contribution `-0.001952`
- `lag_00__CT4__flash_duration`: contribution `-0.001911`
- `lag_02__T3__flash_duration`: contribution `+0.001447`
- `lag_00__T_A_site_active_infernos`: contribution `-0.000808`
