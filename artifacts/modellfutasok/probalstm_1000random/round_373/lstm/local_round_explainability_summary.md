# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `120404`, seconds `0.50`, LSTM `0.9275`, delta `+0.0248`
- tick `124244`, seconds `60.50`, LSTM `0.9426`, delta `+0.0174`
- tick `123796`, seconds `53.50`, LSTM `0.9390`, delta `-0.0164`
- tick `121108`, seconds `11.50`, LSTM `0.9111`, delta `-0.0148`
- tick `124692`, seconds `67.50`, LSTM `0.9599`, delta `+0.0147`
- tick `121268`, seconds `14.00`, LSTM `0.9159`, delta `+0.0146`
- tick `124884`, seconds `70.50`, LSTM `0.9545`, delta `-0.0138`
- tick `121364`, seconds `15.50`, LSTM `0.9308`, delta `+0.0133`
- tick `125684`, seconds `83.00`, LSTM `0.9798`, delta `+0.0117`
- tick `122836`, seconds `38.50`, LSTM `0.9372`, delta `-0.0102`

## Top 15 local ridge features

- `lag_00__CT_place_HEAVEN`: coefficient `-0.000316`, |coef| `0.000316`
- `lag_00__CT_place_CATWALK`: coefficient `0.000307`, |coef| `0.000307`
- `lag_13__CT_place_HELL`: coefficient `-0.000220`, |coef| `0.000220`
- `lag_15__CT_place_HELL`: coefficient `-0.000212`, |coef| `0.000212`
- `lag_12__T_place_LOBBY`: coefficient `0.000205`, |coef| `0.000205`
- `lag_04__CT_place_HELL`: coefficient `0.000198`, |coef| `0.000198`
- `lag_00__CT_kills_last_3s`: coefficient `0.000191`, |coef| `0.000191`
- `lag_11__T_place_CONTROL`: coefficient `0.000188`, |coef| `0.000188`
- `lag_11__CT_place_MINI`: coefficient `0.000176`, |coef| `0.000176`
- `lag_06__T_place_CONTROL`: coefficient `0.000176`, |coef| `0.000176`
- `lag_09__T_place_CONTROL`: coefficient `0.000174`, |coef| `0.000174`
- `lag_09__CT_place_GARAGE`: coefficient `-0.000172`, |coef| `0.000172`
- `lag_15__T_place_LOBBY`: coefficient `0.000168`, |coef| `0.000168`
- `lag_05__T_place_VENDING`: coefficient `0.000168`, |coef| `0.000168`
- `lag_08__T_place_CONTROL`: coefficient `0.000168`, |coef| `0.000168`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_infernos`: coefficient `-0.000162` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000147` (raises CT win probability)
- `lag_01__CT_molly_inv`: coefficient `0.000132` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000132` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000124` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `-0.000106` (lowers CT win probability)
- `lag_01__CT3__molly`: coefficient `0.000105` (raises CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.000102` (lowers CT win probability)
- `lag_01__CT1__molly`: coefficient `0.000092` (raises CT win probability)
- `lag_01__CT5__molly`: coefficient `0.000092` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_HEAVEN`: coefficient `-0.000316` (lowers CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `0.000307` (raises CT win probability)
- `lag_13__CT_place_HELL`: coefficient `-0.000220` (lowers CT win probability)
- `lag_15__CT_place_HELL`: coefficient `-0.000212` (lowers CT win probability)
- `lag_12__T_place_LOBBY`: coefficient `0.000205` (raises CT win probability)
- `lag_04__CT_place_HELL`: coefficient `0.000198` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000191` (raises CT win probability)
- `lag_11__T_place_CONTROL`: coefficient `0.000188` (raises CT win probability)
- `lag_11__CT_place_MINI`: coefficient `0.000176` (raises CT win probability)
- `lag_06__T_place_CONTROL`: coefficient `0.000176` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `120404`, seconds `0.50`, LSTM delta `+0.0248`

Top all feature movements:
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000733`
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000731`
- `lag_01__T_place_TSPAWN`: contribution `+0.000720`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000648`
- `lag_00__T4__duck_amount`: contribution `+0.000597`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `+0.000483`
- `lag_01__smoke_inv_diff`: contribution `+0.000424`
- `lag_01__utility_inv_diff`: contribution `+0.000410`
- `lag_01__CT_molly_inv`: contribution `+0.000395`
- `lag_01__CT_smoke_inv`: contribution `+0.000206`

### tick `124244`, seconds `60.50`, LSTM delta `+0.0174`

Top all feature movements:
- `lag_00__CT_place_HEAVEN`: contribution `+0.001708`
- `lag_00__CT_place_CATWALK`: contribution `+0.001222`
- `lag_13__CT_place_HELL`: contribution `+0.001194`
- `lag_08__T_place_ROOF`: contribution `+0.000874`
- `lag_05__T_place_VENDING`: contribution `+0.000851`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `123796`, seconds `53.50`, LSTM delta `-0.0164`

Top all feature movements:
- `lag_11__CT_place_MINI`: contribution `-0.001081`
- `lag_03__T_place_SILO`: contribution `-0.000987`
- `lag_03__T_place_ROOF`: contribution `-0.000744`
- `lag_02__CT_place_HELL`: contribution `-0.000547`
- `lag_10__CT5__duck_amount`: contribution `-0.000546`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121108`, seconds `11.50`, LSTM delta `-0.0148`

Top all feature movements:
- `lag_13__CT_place_HELL`: contribution `-0.001194`
- `lag_12__CT_place_HELL`: contribution `-0.001188`
- `lag_00__T_place_SILO`: contribution `-0.001010`
- `lag_08__T_place_ROOF`: contribution `-0.000874`
- `lag_02__T_place_SQUEAKY`: contribution `-0.000844`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `-0.000362`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.000214`

### tick `124692`, seconds `67.50`, LSTM delta `+0.0147`

Top all feature movements:
- `lag_00__CT_place_ADMIN`: contribution `-0.001163`
- `lag_03__CT_place_ADMIN`: contribution `+0.000977`
- `lag_15__T_place_SQUEAKY`: contribution `+0.000896`
- `lag_15__T_place_LOBBY`: contribution `+0.000890`
- `lag_14__T_place_VENDING`: contribution `+0.000788`

Top utility-only movements:
- No utility movement among the top local contributors.
