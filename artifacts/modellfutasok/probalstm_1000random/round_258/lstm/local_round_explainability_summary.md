# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `65915`, seconds `109.50`, LSTM `0.5202`, delta `+0.3085`
- tick `65723`, seconds `106.50`, LSTM `0.3680`, delta `-0.1777`
- tick `66267`, seconds `115.00`, LSTM `0.6728`, delta `+0.1620`
- tick `66683`, seconds `121.50`, LSTM `0.7813`, delta `+0.1047`
- tick `66875`, seconds `124.50`, LSTM `0.9435`, delta `+0.0910`
- tick `65563`, seconds `104.00`, LSTM `0.5147`, delta `+0.0799`
- tick `65755`, seconds `107.00`, LSTM `0.3131`, delta `-0.0549`
- tick `65883`, seconds `109.00`, LSTM `0.2117`, delta `-0.0518`
- tick `66299`, seconds `115.50`, LSTM `0.7220`, delta `+0.0492`
- tick `66587`, seconds `120.00`, LSTM `0.6707`, delta `-0.0451`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004478`, |coef| `0.004478`
- `lag_13__T_place_ARCH`: coefficient `-0.004329`, |coef| `0.004329`
- `lag_00__damage_diff_last_5s`: coefficient `0.003375`, |coef| `0.003375`
- `lag_00__CT_defusing_count`: coefficient `0.003085`, |coef| `0.003085`
- `lag_00__T_kills_last_3s`: coefficient `-0.002976`, |coef| `0.002976`
- `lag_00__CT_kills_last_3s`: coefficient `0.002659`, |coef| `0.002659`
- `lag_05__CT_kills_last_3s`: coefficient `-0.002513`, |coef| `0.002513`
- `lag_00__CT_damage_last_5s`: coefficient `0.002353`, |coef| `0.002353`
- `lag_11__CT5__duck_amount`: coefficient `-0.002207`, |coef| `0.002207`
- `lag_09__T2__is_walking`: coefficient `-0.001975`, |coef| `0.001975`
- `lag_07__T_place_ARCH`: coefficient `0.001965`, |coef| `0.001965`
- `lag_14__T2__duck_amount`: coefficient `-0.001908`, |coef| `0.001908`
- `lag_06__CT4__is_scoped`: coefficient `0.001875`, |coef| `0.001875`
- `lag_05__T2__duck_amount`: coefficient `-0.001828`, |coef| `0.001828`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.001671`, |coef| `0.001671`

## Top 10 utility ridge features

- `lag_00__CT1__smoke`: coefficient `0.000869` (raises CT win probability)
- `lag_13__T5__smoke`: coefficient `-0.000660` (lowers CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.000607` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000588` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.000568` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `-0.000565` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000529` (raises CT win probability)
- `lag_03__CT_active_infernos`: coefficient `0.000507` (raises CT win probability)
- `lag_13__T5__utility_total`: coefficient `-0.000506` (lowers CT win probability)
- `lag_14__T5__smoke`: coefficient `-0.000495` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004478` (raises CT win probability)
- `lag_13__T_place_ARCH`: coefficient `-0.004329` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003375` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003085` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002976` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002659` (raises CT win probability)
- `lag_05__CT_kills_last_3s`: coefficient `-0.002513` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002353` (raises CT win probability)
- `lag_11__CT5__duck_amount`: coefficient `-0.002207` (lowers CT win probability)
- `lag_09__T2__is_walking`: coefficient `-0.001975` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65915`, seconds `109.50`, LSTM delta `+0.3085`

Top all feature movements:
- `lag_13__T_place_ARCH`: contribution `+0.040274`
- `lag_00__kill_diff_last_3s`: contribution `+0.021555`
- `lag_00__T_kills_last_3s`: contribution `+0.009429`
- `lag_11__CT5__duck_amount`: contribution `+0.008331`
- `lag_00__CT_kills_last_3s`: contribution `+0.007676`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `65723`, seconds `106.50`, LSTM delta `-0.1777`

Top all feature movements:
- `lag_07__T_place_ARCH`: contribution `-0.018286`
- `lag_00__kill_diff_last_3s`: contribution `-0.010777`
- `lag_00__T_kills_last_3s`: contribution `-0.009429`
- `lag_00__damage_diff_last_5s`: contribution `-0.007614`
- `lag_05__CT_kills_last_3s`: contribution `-0.007256`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66267`, seconds `115.00`, LSTM delta `+0.1620`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.010777`
- `lag_09__CT_place_BALCONY`: contribution `+0.008296`
- `lag_00__CT_kills_last_3s`: contribution `+0.007676`
- `lag_00__damage_diff_last_5s`: contribution `+0.007614`
- `lag_05__CT_kills_last_3s`: contribution `+0.007256`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66683`, seconds `121.50`, LSTM delta `+0.1047`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.029903`
- `lag_15__T_bomb_zone_count`: contribution `+0.007290`
- `lag_15__CT_place_PIT`: contribution `+0.005906`
- `lag_13__CT_place_RUINS`: contribution `+0.005750`
- `lag_09__T2__is_walking`: contribution `+0.004537`

Top utility-only movements:
- `lag_00__CT1__smoke`: contribution `+0.001883`

### tick `66875`, seconds `124.50`, LSTM delta `+0.0910`

Top all feature movements:
- `lag_06__CT_defusing_count`: contribution `+0.013540`
- `lag_00__kill_diff_last_3s`: contribution `+0.010777`
- `lag_01__CT_place_LIBRARY`: contribution `+0.008384`
- `lag_00__CT_kills_last_3s`: contribution `+0.007676`
- `lag_00__damage_diff_last_5s`: contribution `+0.006777`

Top utility-only movements:
- `lag_06__CT1__smoke`: contribution `+0.000972`
