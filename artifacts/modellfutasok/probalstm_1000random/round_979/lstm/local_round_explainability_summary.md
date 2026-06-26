# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `21`

## Largest probability jumps

- tick `182078`, seconds `73.50`, LSTM `0.2845`, delta `-0.2530`
- tick `182302`, seconds `77.00`, LSTM `0.0211`, delta `-0.0936`
- tick `182174`, seconds `75.00`, LSTM `0.1804`, delta `-0.0556`
- tick `182206`, seconds `75.50`, LSTM `0.1455`, delta `-0.0350`
- tick `177502`, seconds `2.00`, LSTM `0.5476`, delta `+0.0329`
- tick `182110`, seconds `74.00`, LSTM `0.2553`, delta `-0.0292`
- tick `179102`, seconds `27.00`, LSTM `0.5565`, delta `-0.0221`
- tick `177982`, seconds `9.50`, LSTM `0.5504`, delta `-0.0198`
- tick `182142`, seconds `74.50`, LSTM `0.2360`, delta `-0.0193`
- tick `179550`, seconds `34.00`, LSTM `0.5542`, delta `-0.0190`

## Top 15 local ridge features

- `lag_15__T_place_CTSPAWN`: coefficient `-0.002831`, |coef| `0.002831`
- `lag_00__CT3__flash`: coefficient `0.002571`, |coef| `0.002571`
- `lag_15__T_place_HOUSE`: coefficient `0.002078`, |coef| `0.002078`
- `lag_05__T_place_CTSPAWN`: coefficient `-0.002030`, |coef| `0.002030`
- `lag_00__CT3__utility_total`: coefficient `0.001994`, |coef| `0.001994`
- `lag_00__T_kills_last_3s`: coefficient `-0.001895`, |coef| `0.001895`
- `lag_02__T5__is_scoped`: coefficient `0.001777`, |coef| `0.001777`
- `lag_00__CT3__alive`: coefficient `0.001688`, |coef| `0.001688`
- `lag_00__CT3__hp`: coefficient `0.001664`, |coef| `0.001664`
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001598`, |coef| `0.001598`
- `lag_00__CT3__armor`: coefficient `0.001596`, |coef| `0.001596`
- `lag_00__CT3__has_defuser`: coefficient `0.001579`, |coef| `0.001579`
- `lag_00__CT3__smoke`: coefficient `0.001541`, |coef| `0.001541`
- `lag_12__T3__duck_amount`: coefficient `-0.001532`, |coef| `0.001532`
- `lag_10__T_place_CTSPAWN`: coefficient `-0.001519`, |coef| `0.001519`

## Top 10 utility ridge features

- `lag_00__CT3__flash`: coefficient `0.002571` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001994` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001598` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001541` (raises CT win probability)
- `lag_13__CT5__smoke`: coefficient `0.001392` (raises CT win probability)
- `lag_06__T4__molly`: coefficient `0.001380` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.001107` (lowers CT win probability)
- `lag_12__CT_A_site_active_smokes`: coefficient `-0.001093` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000992` (raises CT win probability)
- `lag_10__CT_A_site_active_smokes`: coefficient `-0.000987` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_CTSPAWN`: coefficient `-0.002831` (lowers CT win probability)
- `lag_15__T_place_HOUSE`: coefficient `0.002078` (raises CT win probability)
- `lag_05__T_place_CTSPAWN`: coefficient `-0.002030` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001895` (lowers CT win probability)
- `lag_02__T5__is_scoped`: coefficient `0.001777` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001688` (raises CT win probability)
- `lag_00__CT3__hp`: coefficient `0.001664` (raises CT win probability)
- `lag_00__CT3__armor`: coefficient `0.001596` (raises CT win probability)
- `lag_00__CT3__has_defuser`: coefficient `0.001579` (raises CT win probability)
- `lag_12__T3__duck_amount`: coefficient `-0.001532` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `182078`, seconds `73.50`, LSTM delta `-0.2530`

Top all feature movements:
- `lag_15__T_place_CTSPAWN`: contribution `-0.013506`
- `lag_05__T_place_CTSPAWN`: contribution `-0.009683`
- `lag_00__CT3__flash`: contribution `-0.009490`
- `lag_15__T_place_HOUSE`: contribution `-0.009138`
- `lag_02__T5__is_scoped`: contribution `-0.008474`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.009490`
- `lag_00__CT3__utility_total`: contribution `-0.005709`
- `lag_03__T_A_site_active_infernos`: contribution `-0.004757`
- `lag_00__CT3__smoke`: contribution `-0.003408`

### tick `182302`, seconds `77.00`, LSTM delta `-0.0936`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.006137`
- `lag_00__T_kills_last_3s`: contribution `-0.006002`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.005396`
- `lag_00__T_shots_fired_sum`: contribution `-0.004542`
- `lag_00__kill_diff_last_3s`: contribution `-0.003465`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.006137`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002347`
- `lag_07__CT3__flash`: contribution `-0.001735`

### tick `182174`, seconds `75.00`, LSTM delta `-0.0556`

Top all feature movements:
- `lag_08__T_place_CTSPAWN`: contribution `-0.003679`
- `lag_12__CT1__is_walking`: contribution `-0.003032`
- `lag_05__T5__is_scoped`: contribution `-0.002788`
- `lag_03__CT3__flash`: contribution `-0.002666`
- `lag_09__CT1__is_walking`: contribution `-0.002656`

Top utility-only movements:
- `lag_03__CT3__flash`: contribution `-0.002666`
- `lag_03__CT3__utility_total`: contribution `-0.001677`

### tick `182206`, seconds `75.50`, LSTM delta `-0.0350`

Top all feature movements:
- `lag_09__T_place_CTSPAWN`: contribution `-0.002909`
- `lag_04__CT3__flash`: contribution `-0.002383`
- `lag_09__T2__duck_amount`: contribution `+0.002356`
- `lag_07__CT1__is_walking`: contribution `-0.001868`
- `lag_13__T2__duck_amount`: contribution `-0.001612`

Top utility-only movements:
- `lag_04__CT3__flash`: contribution `-0.002383`
- `lag_04__CT3__utility_total`: contribution `-0.001484`
- `lag_01__T_A_site_active_infernos`: contribution `-0.001313`

### tick `177502`, seconds `2.00`, LSTM delta `+0.0329`

Top all feature movements:
- `lag_00__T_he_last_5s`: contribution `+0.006417`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `+0.002846`
- `lag_04__CT_place_MAINHALL`: contribution `+0.002178`
- `lag_04__CT3__flash`: contribution `+0.001946`
- `lag_00__T_place_TUNNEL`: contribution `+0.001916`

Top utility-only movements:
- `lag_00__T_he_last_5s`: contribution `+0.006417`
- `lag_04__CT3__flash`: contribution `+0.001946`
- `lag_04__CT3__utility_total`: contribution `+0.001059`
- `lag_04__CT3__smoke`: contribution `+0.000632`
- `lag_04__T5__molly`: contribution `+0.000562`
