# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `132101`, seconds `32.50`, LSTM `0.2660`, delta `-0.2270`
- tick `132133`, seconds `33.00`, LSTM `0.4247`, delta `+0.1587`
- tick `132517`, seconds `39.00`, LSTM `0.2416`, delta `-0.1256`
- tick `133765`, seconds `58.50`, LSTM `0.0392`, delta `-0.0910`
- tick `132485`, seconds `38.50`, LSTM `0.3672`, delta `-0.0836`
- tick `134917`, seconds `76.50`, LSTM `0.0344`, delta `-0.0743`
- tick `132357`, seconds `36.50`, LSTM `0.4866`, delta `+0.0670`
- tick `132773`, seconds `43.00`, LSTM `0.2061`, delta `-0.0637`
- tick `132229`, seconds `34.50`, LSTM `0.3751`, delta `-0.0618`
- tick `132965`, seconds `46.00`, LSTM `0.2435`, delta `+0.0613`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001625`, |coef| `0.001625`
- `lag_00__T_kills_last_3s`: coefficient `-0.001432`, |coef| `0.001432`
- `lag_01__T1__is_scoped`: coefficient `-0.001275`, |coef| `0.001275`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001268`, |coef| `0.001268`
- `lag_00__CT_place_STAIRS`: coefficient `0.001188`, |coef| `0.001188`
- `lag_01__CT_place_STAIRS`: coefficient `0.001124`, |coef| `0.001124`
- `lag_13__CT_place_TRUCK`: coefficient `0.001093`, |coef| `0.001093`
- `lag_09__T1__is_scoped`: coefficient `0.001090`, |coef| `0.001090`
- `lag_10__T2__duck_amount`: coefficient `-0.001067`, |coef| `0.001067`
- `lag_10__CT_place_STAIRS`: coefficient `0.001046`, |coef| `0.001046`
- `lag_06__CT_place_STAIRS`: coefficient `-0.001042`, |coef| `0.001042`
- `lag_01__T3__is_walking`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_06__CT2__duck_amount`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_03__T_place_PALACEINTERIOR`: coefficient `-0.001026`, |coef| `0.001026`
- `lag_05__T_place_STAIRS`: coefficient `-0.001025`, |coef| `0.001025`

## Top 10 utility ridge features

- `lag_02__CT4__flash_duration`: coefficient `-0.000993` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000837` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000743` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `-0.000738` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.000724` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.000716` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000697` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000691` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000686` (lowers CT win probability)
- `lag_13__CT3__smoke`: coefficient `0.000680` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001625` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001432` (lowers CT win probability)
- `lag_01__T1__is_scoped`: coefficient `-0.001275` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001268` (lowers CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `0.001188` (raises CT win probability)
- `lag_01__CT_place_STAIRS`: coefficient `0.001124` (raises CT win probability)
- `lag_13__CT_place_TRUCK`: coefficient `0.001093` (raises CT win probability)
- `lag_09__T1__is_scoped`: coefficient `0.001090` (raises CT win probability)
- `lag_10__T2__duck_amount`: coefficient `-0.001067` (lowers CT win probability)
- `lag_10__CT_place_STAIRS`: coefficient `0.001046` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `132101`, seconds `32.50`, LSTM delta `-0.2270`

Top all feature movements:
- `lag_01__T1__is_scoped`: contribution `-0.007286`
- `lag_13__CT_place_TRUCK`: contribution `-0.007050`
- `lag_09__T1__is_scoped`: contribution `-0.006226`
- `lag_02__T2__flash_duration`: contribution `-0.004785`
- `lag_11__T_place_CONNECTOR`: contribution `-0.004684`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `-0.004785`
- `lag_12__T3__flash_duration`: contribution `-0.002891`

### tick `132133`, seconds `33.00`, LSTM delta `+0.1587`

Top all feature movements:
- `lag_10__CT_place_STAIRS`: contribution `+0.008139`
- `lag_08__CT_shots_fired_sum`: contribution `+0.004434`
- `lag_12__T_place_CONNECTOR`: contribution `+0.004189`
- `lag_10__T2__duck_amount`: contribution `+0.004079`
- `lag_00__kill_diff_last_3s`: contribution `+0.003911`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.003245`
- `lag_13__T3__flash_duration`: contribution `+0.002039`

### tick `132517`, seconds `39.00`, LSTM delta `-0.1256`

Top all feature movements:
- `lag_01__CT_place_STAIRS`: contribution `-0.008746`
- `lag_06__CT_place_STAIRS`: contribution `-0.008111`
- `lag_00__T_place_JUNGLE`: contribution `-0.007365`
- `lag_11__T_place_CONNECTOR`: contribution `-0.004684`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.004353`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `-0.003603`
- `lag_01__CT5__flash_duration`: contribution `-0.002139`

### tick `133765`, seconds `58.50`, LSTM delta `-0.0910`

Top all feature movements:
- `lag_02__CT4__flash_duration`: contribution `-0.007979`
- `lag_00__T_bomb_zone_count`: contribution `-0.007379`
- `lag_06__CT_flashed_players`: contribution `-0.004738`
- `lag_00__T_kills_last_3s`: contribution `-0.004537`
- `lag_00__kill_diff_last_3s`: contribution `-0.003911`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.007979`
- `lag_06__CT1__flash_duration`: contribution `-0.003860`
- `lag_02__CT_flash_duration_sum`: contribution `-0.003494`
- `lag_06__CT_flash_duration_sum`: contribution `-0.001301`

### tick `132485`, seconds `38.50`, LSTM delta `-0.0836`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `-0.009243`
- `lag_11__T_place_CONNECTOR`: contribution `+0.004684`
- `lag_05__CT_place_STAIRS`: contribution `-0.004429`
- `lag_13__T1__is_scoped`: contribution `-0.003802`
- `lag_09__CT_place_UNDERPASS`: contribution `-0.003586`

Top utility-only movements:
- `lag_14__T2__flash_duration`: contribution `-0.003434`
- `lag_00__CT5__flash_duration`: contribution `-0.001808`
