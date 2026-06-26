# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `14`

## Largest probability jumps

- tick `122446`, seconds `60.00`, LSTM `0.2090`, delta `-0.1976`
- tick `124654`, seconds `94.50`, LSTM `0.0236`, delta `-0.0865`
- tick `122478`, seconds `60.50`, LSTM `0.1575`, delta `-0.0515`
- tick `118638`, seconds `0.50`, LSTM `0.2380`, delta `-0.0499`
- tick `119246`, seconds `10.00`, LSTM `0.3331`, delta `-0.0415`
- tick `121486`, seconds `45.00`, LSTM `0.3778`, delta `+0.0378`
- tick `123854`, seconds `82.00`, LSTM `0.1077`, delta `-0.0360`
- tick `118702`, seconds `1.50`, LSTM `0.2562`, delta `+0.0359`
- tick `121070`, seconds `38.50`, LSTM `0.3742`, delta `+0.0353`
- tick `118830`, seconds `3.50`, LSTM `0.3387`, delta `+0.0348`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002051`, |coef| `0.002051`
- `lag_09__bomb_events_last_5s`: coefficient `-0.002034`, |coef| `0.002034`
- `lag_09__T2__has_bomb`: coefficient `-0.001877`, |coef| `0.001877`
- `lag_00__CT_scoped_count`: coefficient `0.001588`, |coef| `0.001588`
- `lag_09__T4__has_bomb`: coefficient `0.001564`, |coef| `0.001564`
- `lag_00__kill_diff_last_3s`: coefficient `0.001558`, |coef| `0.001558`
- `lag_00__T_damage_last_5s`: coefficient `-0.001548`, |coef| `0.001548`
- `lag_00__CT3__alive`: coefficient `0.001468`, |coef| `0.001468`
- `lag_00__CT3__hp`: coefficient `0.001448`, |coef| `0.001448`
- `lag_00__damage_diff_last_5s`: coefficient `0.001440`, |coef| `0.001440`
- `lag_00__CT3__armor`: coefficient `0.001388`, |coef| `0.001388`
- `lag_00__CT3__has_defuser`: coefficient `0.001382`, |coef| `0.001382`
- `lag_00__CT3__smoke`: coefficient `0.001357`, |coef| `0.001357`
- `lag_07__CT3__is_walking`: coefficient `0.001356`, |coef| `0.001356`
- `lag_10__bomb_events_last_5s`: coefficient `-0.001287`, |coef| `0.001287`

## Top 10 utility ridge features

- `lag_00__CT3__smoke`: coefficient `0.001357` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001142` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001076` (raises CT win probability)
- `lag_01__CT3__smoke`: coefficient `0.001002` (raises CT win probability)
- `lag_01__CT3__utility_total`: coefficient `0.000859` (raises CT win probability)
- `lag_01__CT3__flash`: coefficient `0.000782` (raises CT win probability)
- `lag_02__CT3__smoke`: coefficient `0.000690` (raises CT win probability)
- `lag_02__CT3__utility_total`: coefficient `0.000586` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000561` (raises CT win probability)
- `lag_02__CT3__flash`: coefficient `0.000534` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002051` (lowers CT win probability)
- `lag_09__bomb_events_last_5s`: coefficient `-0.002034` (lowers CT win probability)
- `lag_09__T2__has_bomb`: coefficient `-0.001877` (lowers CT win probability)
- `lag_00__CT_scoped_count`: coefficient `0.001588` (raises CT win probability)
- `lag_09__T4__has_bomb`: coefficient `0.001564` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001558` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001548` (lowers CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001468` (raises CT win probability)
- `lag_00__CT3__hp`: coefficient `0.001448` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001440` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122446`, seconds `60.00`, LSTM delta `-0.1976`

Top all feature movements:
- `lag_09__bomb_events_last_5s`: contribution `-0.008500`
- `lag_00__T_kills_last_3s`: contribution `-0.006498`
- `lag_09__T2__has_bomb`: contribution `-0.005859`
- `lag_09__T4__has_bomb`: contribution `-0.004251`
- `lag_00__kill_diff_last_3s`: contribution `-0.003751`

Top utility-only movements:
- `lag_00__CT3__smoke`: contribution `-0.003001`

### tick `124654`, seconds `94.50`, LSTM delta `-0.0865`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.006498`
- `lag_08__CT_place_WALKWAY`: contribution `-0.004258`
- `lag_00__kill_diff_last_3s`: contribution `-0.003751`
- `lag_15__T_place_RESTROOM`: contribution `-0.003571`
- `lag_09__T_place_CONNECTOR`: contribution `-0.002811`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122478`, seconds `60.50`, LSTM delta `-0.0515`

Top all feature movements:
- `lag_10__bomb_events_last_5s`: contribution `-0.005379`
- `lag_00__bomb_events_last_5s`: contribution `-0.005283`
- `lag_10__T2__has_bomb`: contribution `-0.004007`
- `lag_01__T_kills_last_3s`: contribution `-0.003563`
- `lag_15__T3__duck_amount`: contribution `+0.003390`

Top utility-only movements:
- `lag_01__CT3__smoke`: contribution `-0.002216`

### tick `118638`, seconds `0.50`, LSTM delta `-0.0499`

Top all feature movements:
- `lag_00__T_velocity_mean`: contribution `-0.001983`
- `lag_01__CT3__armor`: contribution `-0.001867`
- `lag_01__T_place_TSPAWN`: contribution `-0.001820`
- `lag_00__CT_velocity_mean`: contribution `-0.001781`
- `lag_01__centroid_distance_xy`: contribution `-0.001064`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000737`
- `lag_01__CT3__utility_total`: contribution `-0.000705`
- `lag_01__CT4__flash`: contribution `-0.000690`
- `lag_01__CT3__smoke`: contribution `-0.000635`
- `lag_00__CT5__smoke`: contribution `-0.000553`

### tick `119246`, seconds `10.00`, LSTM delta `-0.0415`

Top all feature movements:
- `lag_14__CT_place_BACKOFA`: contribution `-0.005346`
- `lag_08__T_place_TSTAIRS`: contribution `-0.004548`
- `lag_10__CT_place_BACKOFA`: contribution `+0.004111`
- `lag_00__CT_place_BRIDGE`: contribution `-0.003494`
- `lag_00__CT_place_WALKWAY`: contribution `+0.003431`

Top utility-only movements:
- `lag_04__CT_smokes_last_5s`: contribution `-0.002379`
