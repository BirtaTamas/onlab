# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `134071`, seconds `25.00`, LSTM `0.1234`, delta `-0.1440`
- tick `133847`, seconds `21.50`, LSTM `0.1379`, delta `-0.0736`
- tick `134231`, seconds `27.50`, LSTM `0.0233`, delta `-0.0721`
- tick `132503`, seconds `0.50`, LSTM `0.1477`, delta `-0.0590`
- tick `133143`, seconds `10.50`, LSTM `0.2222`, delta `+0.0548`
- tick `132535`, seconds `1.00`, LSTM `0.2015`, delta `+0.0538`
- tick `133943`, seconds `23.00`, LSTM `0.2269`, delta `+0.0536`
- tick `133303`, seconds `13.00`, LSTM `0.2244`, delta `-0.0487`
- tick `132983`, seconds `8.00`, LSTM `0.2195`, delta `+0.0376`
- tick `133047`, seconds `9.00`, LSTM `0.1719`, delta `-0.0369`

## Top 15 local ridge features

- `lag_10__CT_place_JUNGLE`: coefficient `0.001139`, |coef| `0.001139`
- `lag_14__T_flashes_last_5s`: coefficient `0.001115`, |coef| `0.001115`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001051`, |coef| `0.001051`
- `lag_14__CT_smokes_last_5s`: coefficient `0.001030`, |coef| `0.001030`
- `lag_00__CT_smokes_last_5s`: coefficient `0.000992`, |coef| `0.000992`
- `lag_02__CT3__is_scoped`: coefficient `-0.000975`, |coef| `0.000975`
- `lag_06__T_shots_fired_sum`: coefficient `0.000852`, |coef| `0.000852`
- `lag_02__T_place_LADDER`: coefficient `-0.000828`, |coef| `0.000828`
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.000783`, |coef| `0.000783`
- `lag_10__CT_place_STAIRS`: coefficient `0.000686`, |coef| `0.000686`
- `lag_00__CT_place_STAIRS`: coefficient `0.000682`, |coef| `0.000682`
- `lag_06__CT_smokes_last_5s`: coefficient `0.000658`, |coef| `0.000658`
- `lag_07__T_shots_fired_sum`: coefficient `-0.000632`, |coef| `0.000632`
- `lag_00__CT_place_JUNGLE`: coefficient `0.000607`, |coef| `0.000607`
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000600`, |coef| `0.000600`

## Top 10 utility ridge features

- `lag_14__T_flashes_last_5s`: coefficient `0.001115` (raises CT win probability)
- `lag_14__CT_smokes_last_5s`: coefficient `0.001030` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000992` (raises CT win probability)
- `lag_06__CT_smokes_last_5s`: coefficient `0.000658` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000600` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000583` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000566` (raises CT win probability)
- `lag_09__T3__flash_duration`: coefficient `-0.000549` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `0.000545` (raises CT win probability)
- `lag_15__T_flashes_last_5s`: coefficient `0.000506` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_JUNGLE`: coefficient `0.001139` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001051` (lowers CT win probability)
- `lag_02__CT3__is_scoped`: coefficient `-0.000975` (lowers CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.000852` (raises CT win probability)
- `lag_02__T_place_LADDER`: coefficient `-0.000828` (lowers CT win probability)
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.000783` (lowers CT win probability)
- `lag_10__CT_place_STAIRS`: coefficient `0.000686` (raises CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `0.000682` (raises CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `-0.000632` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.000607` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `134071`, seconds `25.00`, LSTM delta `-0.1440`

Top all feature movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.010106`
- `lag_06__T_shots_fired_sum`: contribution `-0.007668`
- `lag_10__CT_place_JUNGLE`: contribution `-0.007310`
- `lag_07__T_shots_fired_sum`: contribution `-0.005215`
- `lag_00__T_shots_fired_sum`: contribution `-0.004727`

Top utility-only movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.010106`
- `lag_01__T3__flash_duration`: contribution `-0.002740`
- `lag_09__T3__flash_duration`: contribution `-0.002580`
- `lag_01__T_B_site_active_infernos`: contribution `-0.001599`

### tick `133847`, seconds `21.50`, LSTM delta `-0.0736`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.008666`
- `lag_10__CT_place_STAIRS`: contribution `-0.005343`
- `lag_02__CT3__is_scoped`: contribution `-0.004436`
- `lag_07__T_flashes_last_5s`: contribution `-0.003999`
- `lag_07__CT_place_JUNGLE`: contribution `-0.002816`

Top utility-only movements:
- `lag_07__T_flashes_last_5s`: contribution `-0.003999`
- `lag_08__T_B_site_active_infernos`: contribution `-0.001044`
- `lag_02__T3__flash_duration`: contribution `-0.000713`

### tick `134231`, seconds `27.50`, LSTM delta `-0.0721`

Top all feature movements:
- `lag_02__T_place_LADDER`: contribution `-0.018716`
- `lag_00__T_shots_fired_sum`: contribution `-0.004727`
- `lag_06__CT3__is_scoped`: contribution `-0.002199`
- `lag_15__CT_place_JUNGLE`: contribution `-0.001967`
- `lag_14__T3__flash_duration`: contribution `-0.001886`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `-0.001886`
- `lag_06__T3__flash_duration`: contribution `-0.001686`

### tick `132503`, seconds `0.50`, LSTM delta `-0.0590`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002054`
- `lag_01__T_place_TSPAWN`: contribution `-0.001975`
- `lag_00__CT_velocity_mean`: contribution `-0.001777`
- `lag_00__T_velocity_mean`: contribution `-0.001166`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000945`

Top utility-only movements:
- `lag_00__T1__smoke`: contribution `-0.000681`
- `lag_01__T5__flash`: contribution `-0.000500`
- `lag_01__T3__molly`: contribution `-0.000431`
- `lag_01__utility_inv_diff`: contribution `-0.000431`
- `lag_01__smoke_inv_diff`: contribution `-0.000402`

### tick `133143`, seconds `10.50`, LSTM delta `+0.0548`

Top all feature movements:
- `lag_09__CT_smokes_last_5s`: contribution `+0.007097`
- `lag_00__CT_place_STAIRS`: contribution `+0.005305`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.002675`
- `lag_06__CT3__is_scoped`: contribution `+0.002199`
- `lag_05__CT3__is_scoped`: contribution `-0.002192`

Top utility-only movements:
- `lag_09__CT_smokes_last_5s`: contribution `+0.007097`
- `lag_03__T3__flash_duration`: contribution `+0.000605`
- `lag_01__T3__molly`: contribution `+0.000601`
