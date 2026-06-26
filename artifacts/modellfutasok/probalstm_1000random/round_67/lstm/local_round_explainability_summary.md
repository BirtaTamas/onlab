# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `60437`, seconds `59.50`, LSTM `0.0387`, delta `-0.1724`
- tick `60405`, seconds `59.00`, LSTM `0.2111`, delta `+0.1445`
- tick `62101`, seconds `85.50`, LSTM `0.2377`, delta `-0.1258`
- tick `59765`, seconds `49.00`, LSTM `0.0725`, delta `-0.1195`
- tick `62485`, seconds `91.50`, LSTM `0.0474`, delta `-0.0912`
- tick `62005`, seconds `84.00`, LSTM `0.2907`, delta `+0.0761`
- tick `56661`, seconds `0.50`, LSTM `0.1767`, delta `-0.0685`
- tick `58005`, seconds `21.50`, LSTM `0.2833`, delta `+0.0597`
- tick `59669`, seconds `47.50`, LSTM `0.2428`, delta `-0.0555`
- tick `60213`, seconds `56.00`, LSTM `0.1168`, delta `+0.0541`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002199`, |coef| `0.002199`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002121`, |coef| `0.002121`
- `lag_10__T_place_QUAD`: coefficient `-0.001972`, |coef| `0.001972`
- `lag_09__T_place_QUAD`: coefficient `0.001832`, |coef| `0.001832`
- `lag_00__T_place_SECONDMID`: coefficient `0.001713`, |coef| `0.001713`
- `lag_00__kill_diff_last_3s`: coefficient `0.001664`, |coef| `0.001664`
- `lag_14__CT_place_LOWERMID`: coefficient `0.001410`, |coef| `0.001410`
- `lag_13__T2__duck_amount`: coefficient `0.001364`, |coef| `0.001364`
- `lag_00__CT5__alive`: coefficient `0.001288`, |coef| `0.001288`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001274`, |coef| `0.001274`
- `lag_00__T_damage_last_5s`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_07__CT_flashes_last_5s`: coefficient `0.001244`, |coef| `0.001244`
- `lag_02__T_place_SECONDMID`: coefficient `0.001201`, |coef| `0.001201`
- `lag_00__CT5__armor`: coefficient `0.001196`, |coef| `0.001196`
- `lag_10__T1__is_walking`: coefficient `0.001188`, |coef| `0.001188`

## Top 10 utility ridge features

- `lag_07__CT_flashes_last_5s`: coefficient `0.001244` (raises CT win probability)
- `lag_02__T1__molly`: coefficient `-0.001122` (lowers CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.001120` (raises CT win probability)
- `lag_04__T5__molly`: coefficient `0.001116` (raises CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `0.001074` (raises CT win probability)
- `lag_03__T1__molly`: coefficient `-0.000966` (lowers CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000925` (lowers CT win probability)
- `lag_03__T5__molly`: coefficient `0.000916` (raises CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `0.000857` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000837` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002199` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002121` (lowers CT win probability)
- `lag_10__T_place_QUAD`: coefficient `-0.001972` (lowers CT win probability)
- `lag_09__T_place_QUAD`: coefficient `0.001832` (raises CT win probability)
- `lag_00__T_place_SECONDMID`: coefficient `0.001713` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001664` (raises CT win probability)
- `lag_14__CT_place_LOWERMID`: coefficient `0.001410` (raises CT win probability)
- `lag_13__T2__duck_amount`: coefficient `0.001364` (raises CT win probability)
- `lag_00__CT5__alive`: coefficient `0.001288` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001274` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `60437`, seconds `59.50`, LSTM delta `-0.1724`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `-0.095009`
- `lag_05__T_place_BALCONY`: contribution `-0.015437`
- `lag_05__T_place_QUAD`: contribution `+0.012021`
- `lag_08__T_place_QUAD`: contribution `-0.009198`
- `lag_00__T_kills_last_3s`: contribution `-0.006966`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.001122`

### tick `60405`, seconds `59.00`, LSTM delta `+0.1445`

Top all feature movements:
- `lag_09__T_place_QUAD`: contribution `+0.088248`
- `lag_04__T_place_QUAD`: contribution `+0.014159`
- `lag_07__T_place_QUAD`: contribution `+0.013539`
- `lag_04__T_place_BALCONY`: contribution `+0.007337`
- `lag_00__T_damage_last_5s`: contribution `-0.002379`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `+0.001037`

### tick `62101`, seconds `85.50`, LSTM delta `-0.1258`

Top all feature movements:
- `lag_14__CT_place_LOWERMID`: contribution `-0.038690`
- `lag_09__CT_place_TRAMP`: contribution `-0.015202`
- `lag_07__CT_flashes_last_5s`: contribution `-0.013681`
- `lag_00__T_kills_last_3s`: contribution `-0.006966`
- `lag_00__T_shots_fired_sum`: contribution `+0.006361`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.013681`

### tick `59765`, seconds `49.00`, LSTM delta `-0.1195`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.007951`
- `lag_00__T_kills_last_3s`: contribution `-0.006966`
- `lag_13__T2__duck_amount`: contribution `-0.005214`
- `lag_00__kill_diff_last_3s`: contribution `-0.004004`
- `lag_08__T4__duck_amount`: contribution `-0.003949`

Top utility-only movements:
- `lag_04__T5__molly`: contribution `-0.002469`
- `lag_03__T1__molly`: contribution `-0.002139`

### tick `62485`, seconds `91.50`, LSTM delta `-0.0912`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.007951`
- `lag_12__CT_place_QUAD`: contribution `-0.007676`
- `lag_00__T_kills_last_3s`: contribution `-0.006966`
- `lag_10__CT_place_QUAD`: contribution `-0.006505`
- `lag_01__T_shots_fired_sum`: contribution `-0.004775`

Top utility-only movements:
- No utility movement among the top local contributors.
