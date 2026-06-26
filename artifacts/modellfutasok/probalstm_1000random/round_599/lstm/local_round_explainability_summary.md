# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `55415`, seconds `28.00`, LSTM `0.7557`, delta `+0.1340`
- tick `57719`, seconds `64.00`, LSTM `0.9213`, delta `+0.1235`
- tick `54295`, seconds `10.50`, LSTM `0.5116`, delta `+0.0516`
- tick `55639`, seconds `31.50`, LSTM `0.6883`, delta `-0.0513`
- tick `55383`, seconds `27.50`, LSTM `0.6217`, delta `+0.0504`
- tick `55447`, seconds `28.50`, LSTM `0.7946`, delta `+0.0389`
- tick `60759`, seconds `111.50`, LSTM `0.9237`, delta `+0.0378`
- tick `61687`, seconds `126.00`, LSTM `0.9081`, delta `+0.0353`
- tick `55607`, seconds `31.00`, LSTM `0.7396`, delta `-0.0323`
- tick `56663`, seconds `47.50`, LSTM `0.7967`, delta `+0.0308`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002686`, |coef| `0.002686`
- `lag_01__CT_place_RAFTERS`: coefficient `-0.002239`, |coef| `0.002239`
- `lag_01__CT_place_HEAVEN`: coefficient `0.002139`, |coef| `0.002139`
- `lag_00__T5__flash`: coefficient `-0.002091`, |coef| `0.002091`
- `lag_00__kill_diff_last_3s`: coefficient `0.002065`, |coef| `0.002065`
- `lag_00__T5__is_scoped`: coefficient `0.001967`, |coef| `0.001967`
- `lag_00__T1__smoke`: coefficient `-0.001917`, |coef| `0.001917`
- `lag_00__CT_damage_last_5s`: coefficient `0.001914`, |coef| `0.001914`
- `lag_15__T5__duck_amount`: coefficient `-0.001882`, |coef| `0.001882`
- `lag_00__T5__utility_total`: coefficient `-0.001864`, |coef| `0.001864`
- `lag_00__T5__alive`: coefficient `-0.001799`, |coef| `0.001799`
- `lag_00__damage_diff_last_5s`: coefficient `0.001796`, |coef| `0.001796`
- `lag_00__T5__hp`: coefficient `-0.001768`, |coef| `0.001768`
- `lag_00__CT_place_HUT`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_00__T5__armor`: coefficient `-0.001633`, |coef| `0.001633`

## Top 10 utility ridge features

- `lag_00__T5__flash`: coefficient `-0.002091` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001917` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001864` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.001630` (lowers CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.001392` (raises CT win probability)
- `lag_12__T1__smoke`: coefficient `-0.001390` (lowers CT win probability)
- `lag_07__T1__smoke`: coefficient `-0.001126` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.000986` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000950` (raises CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `0.000911` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002686` (raises CT win probability)
- `lag_01__CT_place_RAFTERS`: coefficient `-0.002239` (lowers CT win probability)
- `lag_01__CT_place_HEAVEN`: coefficient `0.002139` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002065` (raises CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.001967` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001914` (raises CT win probability)
- `lag_15__T5__duck_amount`: coefficient `-0.001882` (lowers CT win probability)
- `lag_00__T5__alive`: coefficient `-0.001799` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001796` (raises CT win probability)
- `lag_00__T5__hp`: coefficient `-0.001768` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `55415`, seconds `28.00`, LSTM delta `+0.1340`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `+0.009381`
- `lag_08__CT5__flash_duration`: contribution `+0.008864`
- `lag_00__CT_kills_last_3s`: contribution `+0.007755`
- `lag_00__kill_diff_last_3s`: contribution `+0.004970`
- `lag_02__T5__is_scoped`: contribution `+0.004060`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.008864`
- `lag_08__CT_flash_duration_sum`: contribution `+0.003963`
- `lag_03__CT4__flash_duration`: contribution `+0.003119`
- `lag_08__CT4__flash_duration`: contribution `+0.002592`

### tick `57719`, seconds `64.00`, LSTM delta `+0.1235`

Top all feature movements:
- `lag_01__CT_place_RAFTERS`: contribution `+0.011966`
- `lag_01__CT_place_HEAVEN`: contribution `+0.011551`
- `lag_00__CT_kills_last_3s`: contribution `+0.007755`
- `lag_15__T5__duck_amount`: contribution `+0.007145`
- `lag_00__T5__flash`: contribution `+0.005935`

Top utility-only movements:
- `lag_00__T5__flash`: contribution `+0.005935`
- `lag_00__T5__utility_total`: contribution `+0.004474`
- `lag_00__T5__molly`: contribution `+0.003606`
- `lag_12__T1__smoke`: contribution `+0.003000`

### tick `54295`, seconds `10.50`, LSTM delta `+0.0516`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `+0.015230`
- `lag_07__CT_place_HELL`: contribution `+0.004888`
- `lag_07__T_place_ROOF`: contribution `+0.004153`
- `lag_04__CT_place_RAFTERS`: contribution `+0.003707`
- `lag_00__CT_place_RAFTERS`: contribution `-0.003571`

Top utility-only movements:
- `lag_00__T5__utility_total`: contribution `+0.001491`
- `lag_03__CT2__molly`: contribution `+0.001447`
- `lag_03__CT5__flash_duration`: contribution `+0.001230`

### tick `55639`, seconds `31.50`, LSTM delta `-0.0513`

Top all feature movements:
- `lag_00__CT_damage_last_5s`: contribution `-0.004130`
- `lag_00__damage_diff_last_5s`: contribution `-0.004012`
- `lag_04__CT5__flash_duration`: contribution `-0.003378`
- `lag_08__CT5__duck_amount`: contribution `-0.003241`
- `lag_08__CT4__duck_amount`: contribution `-0.003179`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `-0.003378`
- `lag_15__CT5__flash_duration`: contribution `-0.001870`

### tick `55383`, seconds `27.50`, LSTM delta `+0.0504`

Top all feature movements:
- `lag_07__CT5__flash_duration`: contribution `+0.006046`
- `lag_00__CT4__duck_amount`: contribution `+0.002871`
- `lag_02__CT1__shots_fired`: contribution `+0.002845`
- `lag_15__CT4__duck_amount`: contribution `+0.002642`
- `lag_07__CT_flashed_players`: contribution `+0.002559`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.006046`
- `lag_07__CT_flash_duration_sum`: contribution `+0.002558`
- `lag_07__CT4__flash_duration`: contribution `+0.001460`
