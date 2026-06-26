# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `16`

## Largest probability jumps

- tick `137689`, seconds `32.50`, LSTM `0.6570`, delta `+0.1522`
- tick `137849`, seconds `35.00`, LSTM `0.8802`, delta `+0.1367`
- tick `139289`, seconds `57.50`, LSTM `0.8919`, delta `+0.1355`
- tick `139481`, seconds `60.50`, LSTM `0.9576`, delta `+0.0920`
- tick `139385`, seconds `59.00`, LSTM `0.8405`, delta `-0.0636`
- tick `137721`, seconds `33.00`, LSTM `0.6958`, delta `+0.0388`
- tick `138009`, seconds `37.50`, LSTM `0.8413`, delta `-0.0343`
- tick `136345`, seconds `11.50`, LSTM `0.4857`, delta `-0.0278`
- tick `136665`, seconds `16.50`, LSTM `0.5185`, delta `+0.0233`
- tick `139097`, seconds `54.50`, LSTM `0.7835`, delta `-0.0226`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001587`, |coef| `0.001587`
- `lag_14__T_place_IVY`: coefficient `-0.001566`, |coef| `0.001566`
- `lag_00__kill_diff_last_3s`: coefficient `0.001520`, |coef| `0.001520`
- `lag_07__CT_place_BACKOFB`: coefficient `0.001462`, |coef| `0.001462`
- `lag_14__T5__is_scoped`: coefficient `-0.001333`, |coef| `0.001333`
- `lag_00__CT_damage_last_5s`: coefficient `0.001317`, |coef| `0.001317`
- `lag_00__damage_diff_last_5s`: coefficient `0.001277`, |coef| `0.001277`
- `lag_08__T_place_IVY`: coefficient `-0.001236`, |coef| `0.001236`
- `lag_15__T_place_IVY`: coefficient `-0.001165`, |coef| `0.001165`
- `lag_04__T_place_TSTAIRS`: coefficient `0.000975`, |coef| `0.000975`
- `lag_09__T_place_TSTAIRS`: coefficient `0.000967`, |coef| `0.000967`
- `lag_03__CT3__is_scoped`: coefficient `0.000949`, |coef| `0.000949`
- `lag_00__CT_place_LONGDOG`: coefficient `0.000920`, |coef| `0.000920`
- `lag_00__CT3__is_scoped`: coefficient `-0.000898`, |coef| `0.000898`
- `lag_14__T5__is_walking`: coefficient `0.000897`, |coef| `0.000897`

## Top 10 utility ridge features

- `lag_08__CT_A_site_active_infernos`: coefficient `0.000700` (raises CT win probability)
- `lag_06__T_A_site_active_smokes`: coefficient `-0.000618` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000606` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000594` (lowers CT win probability)
- `lag_10__CT4__molly`: coefficient `-0.000525` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000517` (lowers CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000515` (lowers CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `-0.000485` (lowers CT win probability)
- `lag_06__T_active_smokes`: coefficient `-0.000484` (lowers CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `-0.000484` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001587` (raises CT win probability)
- `lag_14__T_place_IVY`: coefficient `-0.001566` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001520` (raises CT win probability)
- `lag_07__CT_place_BACKOFB`: coefficient `0.001462` (raises CT win probability)
- `lag_14__T5__is_scoped`: coefficient `-0.001333` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001317` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001277` (raises CT win probability)
- `lag_08__T_place_IVY`: coefficient `-0.001236` (lowers CT win probability)
- `lag_15__T_place_IVY`: coefficient `-0.001165` (lowers CT win probability)
- `lag_04__T_place_TSTAIRS`: coefficient `0.000975` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `137689`, seconds `32.50`, LSTM delta `+0.1522`

Top all feature movements:
- `lag_14__T_place_IVY`: contribution `+0.008367`
- `lag_14__T5__is_scoped`: contribution `+0.006357`
- `lag_04__T_place_TSTAIRS`: contribution `+0.005526`
- `lag_00__CT_kills_last_3s`: contribution `+0.004582`
- `lag_10__T_place_IVY`: contribution `+0.004180`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.002470`

### tick `137849`, seconds `35.00`, LSTM delta `+0.1367`

Top all feature movements:
- `lag_15__T_place_IVY`: contribution `+0.006226`
- `lag_09__T_place_TSTAIRS`: contribution `+0.005481`
- `lag_00__CT_kills_last_3s`: contribution `+0.004582`
- `lag_00__kill_diff_last_3s`: contribution `+0.003660`
- `lag_00__T5__is_scoped`: contribution `+0.002893`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `+0.001593`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.001523`

### tick `139289`, seconds `57.50`, LSTM delta `+0.1355`

Top all feature movements:
- `lag_07__CT_place_BACKOFB`: contribution `+0.008349`
- `lag_08__T_place_IVY`: contribution `+0.006602`
- `lag_14__T5__is_scoped`: contribution `+0.006357`
- `lag_00__CT_kills_last_3s`: contribution `+0.004582`
- `lag_00__CT3__is_scoped`: contribution `+0.004085`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139481`, seconds `60.50`, LSTM delta `+0.0920`

Top all feature movements:
- `lag_14__T_place_IVY`: contribution `+0.008367`
- `lag_03__CT3__is_scoped`: contribution `+0.004316`
- `lag_00__T_place_LONGDOG`: contribution `+0.004000`
- `lag_13__CT_place_BACKOFB`: contribution `+0.003349`
- `lag_00__CT_damage_last_5s`: contribution `+0.002154`

Top utility-only movements:
- `lag_02__CT_A_site_active_smokes`: contribution `+0.001483`

### tick `139385`, seconds `59.00`, LSTM delta `-0.0636`

Top all feature movements:
- `lag_00__CT_place_LONGDOG`: contribution `-0.006002`
- `lag_03__CT3__is_scoped`: contribution `-0.004316`
- `lag_00__CT3__is_scoped`: contribution `-0.004085`
- `lag_00__kill_diff_last_3s`: contribution `-0.003660`
- `lag_00__damage_diff_last_5s`: contribution `-0.002882`

Top utility-only movements:
- No utility movement among the top local contributors.
