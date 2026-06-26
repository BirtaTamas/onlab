# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `11`

## Largest probability jumps

- tick `76459`, seconds `93.00`, LSTM `0.5423`, delta `+0.2993`
- tick `72459`, seconds `30.50`, LSTM `0.4797`, delta `-0.2151`
- tick `72395`, seconds `29.50`, LSTM `0.7341`, delta `+0.1884`
- tick `74699`, seconds `65.50`, LSTM `0.2277`, delta `-0.1741`
- tick `76715`, seconds `97.00`, LSTM `0.8950`, delta `+0.1448`
- tick `72363`, seconds `29.00`, LSTM `0.5458`, delta `-0.1313`
- tick `76619`, seconds `95.50`, LSTM `0.7697`, delta `+0.1118`
- tick `72171`, seconds `26.00`, LSTM `0.6387`, delta `+0.1008`
- tick `74731`, seconds `66.00`, LSTM `0.1286`, delta `-0.0990`
- tick `76587`, seconds `95.00`, LSTM `0.6580`, delta `+0.0914`

## Top 15 local ridge features

- `lag_04__T_place_MINI`: coefficient `-0.004314`, |coef| `0.004314`
- `lag_13__CT_place_SQUEAKY`: coefficient `0.004003`, |coef| `0.004003`
- `lag_08__T_place_HUT`: coefficient `-0.003883`, |coef| `0.003883`
- `lag_07__CT_place_SQUEAKY`: coefficient `-0.003717`, |coef| `0.003717`
- `lag_00__T_place_RAFTERS`: coefficient `-0.002885`, |coef| `0.002885`
- `lag_00__kill_diff_last_3s`: coefficient `0.002709`, |coef| `0.002709`
- `lag_05__T_place_MINI`: coefficient `-0.002303`, |coef| `0.002303`
- `lag_00__CT_kills_last_3s`: coefficient `0.002188`, |coef| `0.002188`
- `lag_03__T_place_MINI`: coefficient `-0.002101`, |coef| `0.002101`
- `lag_00__CT_defusing_count`: coefficient `0.001900`, |coef| `0.001900`
- `lag_02__T_place_HUT`: coefficient `-0.001893`, |coef| `0.001893`
- `lag_12__T_place_HUT`: coefficient `-0.001826`, |coef| `0.001826`
- `lag_03__T_place_HUT`: coefficient `-0.001804`, |coef| `0.001804`
- `lag_00__T_place_HUT`: coefficient `-0.001782`, |coef| `0.001782`
- `lag_05__CT_place_HEAVEN`: coefficient `0.001680`, |coef| `0.001680`

## Top 10 utility ridge features

- `lag_02__T3__flash_duration`: coefficient `0.001626` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001445` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001421` (raises CT win probability)
- `lag_05__T4__flash_duration`: coefficient `0.001303` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001227` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001183` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.001040` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001021` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `-0.001013` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000960` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_MINI`: coefficient `-0.004314` (lowers CT win probability)
- `lag_13__CT_place_SQUEAKY`: coefficient `0.004003` (raises CT win probability)
- `lag_08__T_place_HUT`: coefficient `-0.003883` (lowers CT win probability)
- `lag_07__CT_place_SQUEAKY`: coefficient `-0.003717` (lowers CT win probability)
- `lag_00__T_place_RAFTERS`: coefficient `-0.002885` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002709` (raises CT win probability)
- `lag_05__T_place_MINI`: coefficient `-0.002303` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002188` (raises CT win probability)
- `lag_03__T_place_MINI`: coefficient `-0.002101` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.001900` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76459`, seconds `93.00`, LSTM delta `+0.2993`

Top all feature movements:
- `lag_13__CT_place_SQUEAKY`: contribution `+0.053231`
- `lag_07__CT_place_SQUEAKY`: contribution `+0.049426`
- `lag_08__T_place_HUT`: contribution `+0.036193`
- `lag_05__CT_place_HEAVEN`: contribution `+0.009071`
- `lag_02__T4__is_scoped`: contribution `+0.007266`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.003670`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.003479`

### tick `72459`, seconds `30.50`, LSTM delta `-0.2151`

Top all feature movements:
- `lag_02__T3__flash_duration`: contribution `-0.010463`
- `lag_00__CT_place_GARAGE`: contribution `-0.009860`
- `lag_07__CT_place_ADMIN`: contribution `-0.009355`
- `lag_04__T_place_SILO`: contribution `-0.008104`
- `lag_11__T_flashed_players`: contribution `-0.007794`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `-0.010463`
- `lag_11__T3__flash_duration`: contribution `-0.006176`
- `lag_11__T_flash_duration_sum`: contribution `-0.003075`

### tick `72395`, seconds `29.50`, LSTM delta `+0.1884`

Top all feature movements:
- `lag_09__T_flashed_players`: contribution `+0.010617`
- `lag_00__kill_diff_last_3s`: contribution `+0.006520`
- `lag_00__CT_kills_last_3s`: contribution `+0.006318`
- `lag_09__T3__flash_duration`: contribution `+0.004629`
- `lag_02__T_place_SILO`: contribution `+0.004579`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `+0.004629`
- `lag_09__T_flash_duration_sum`: contribution `+0.003464`
- `lag_00__T3__flash_duration`: contribution `+0.003006`
- `lag_15__CT4__flash_duration`: contribution `+0.002654`

### tick `74699`, seconds `65.50`, LSTM delta `-0.1741`

Top all feature movements:
- `lag_04__T_place_MINI`: contribution `-0.060014`
- `lag_00__kill_diff_last_3s`: contribution `-0.006520`
- `lag_04__T2__duck_amount`: contribution `-0.004042`
- `lag_00__T_kills_last_3s`: contribution `-0.003687`
- `lag_01__T4__is_scoped`: contribution `-0.003089`

Top utility-only movements:
- `lag_01__T4__smoke`: contribution `-0.001672`

### tick `76715`, seconds `97.00`, LSTM delta `+0.1448`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.018416`
- `lag_15__CT_place_SQUEAKY`: contribution `+0.010522`
- `lag_05__T4__flash_duration`: contribution `+0.010180`
- `lag_03__T_flash_alpha_mean`: contribution `+0.007176`
- `lag_13__CT_place_HEAVEN`: contribution `+0.005889`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.010180`
- `lag_03__T_flash_alpha_mean`: contribution `+0.007176`
