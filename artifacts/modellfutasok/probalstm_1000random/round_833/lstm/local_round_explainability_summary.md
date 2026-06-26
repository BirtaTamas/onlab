# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `19`

## Largest probability jumps

- tick `162340`, seconds `66.50`, LSTM `0.1353`, delta `-0.3457`
- tick `162244`, seconds `65.00`, LSTM `0.3594`, delta `-0.1487`
- tick `162276`, seconds `65.50`, LSTM `0.4889`, delta `+0.1295`
- tick `159908`, seconds `28.50`, LSTM `0.4733`, delta `+0.1031`
- tick `159812`, seconds `27.00`, LSTM `0.4640`, delta `-0.1009`
- tick `159844`, seconds `27.50`, LSTM `0.3904`, delta `-0.0736`
- tick `162436`, seconds `68.00`, LSTM `0.1209`, delta `+0.0609`
- tick `159268`, seconds `18.50`, LSTM `0.5077`, delta `+0.0581`
- tick `158148`, seconds `1.00`, LSTM `0.3618`, delta `-0.0545`
- tick `162372`, seconds `67.00`, LSTM `0.0818`, delta `-0.0534`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.003107`, |coef| `0.003107`
- `lag_00__T_kills_last_3s`: coefficient `-0.003096`, |coef| `0.003096`
- `lag_00__kill_diff_last_3s`: coefficient `0.002470`, |coef| `0.002470`
- `lag_04__T_place_MAINHALL`: coefficient `0.002249`, |coef| `0.002249`
- `lag_04__CT3__shots_fired`: coefficient `-0.002182`, |coef| `0.002182`
- `lag_05__CT3__duck_amount`: coefficient `-0.001957`, |coef| `0.001957`
- `lag_02__CT_kills_last_3s`: coefficient `-0.001747`, |coef| `0.001747`
- `lag_02__T5__duck_amount`: coefficient `0.001708`, |coef| `0.001708`
- `lag_00__CT1__alive`: coefficient `0.001708`, |coef| `0.001708`
- `lag_00__CT1__hp`: coefficient `0.001684`, |coef| `0.001684`
- `lag_04__T5__duck_amount`: coefficient `-0.001681`, |coef| `0.001681`
- `lag_05__T_place_MAINHALL`: coefficient `0.001654`, |coef| `0.001654`
- `lag_03__CT3__alive`: coefficient `0.001654`, |coef| `0.001654`
- `lag_02__T5__alive`: coefficient `0.001612`, |coef| `0.001612`
- `lag_05__CT3__shots_fired`: coefficient `-0.001608`, |coef| `0.001608`

## Top 10 utility ridge features

- `lag_10__CT3__smoke`: coefficient `0.001409` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001322` (lowers CT win probability)
- `lag_02__T5__smoke`: coefficient `0.001304` (raises CT win probability)
- `lag_03__CT3__flash`: coefficient `0.001276` (raises CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `-0.000970` (lowers CT win probability)
- `lag_03__CT3__utility_total`: coefficient `0.000776` (raises CT win probability)
- `lag_10__CT3__utility_total`: coefficient `0.000739` (raises CT win probability)
- `lag_01__T5__smoke`: coefficient `0.000688` (raises CT win probability)
- `lag_11__CT3__smoke`: coefficient `0.000656` (raises CT win probability)
- `lag_07__CT3__smoke`: coefficient `0.000647` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.003107` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003096` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002470` (raises CT win probability)
- `lag_04__T_place_MAINHALL`: coefficient `0.002249` (raises CT win probability)
- `lag_04__CT3__shots_fired`: coefficient `-0.002182` (lowers CT win probability)
- `lag_05__CT3__duck_amount`: coefficient `-0.001957` (lowers CT win probability)
- `lag_02__CT_kills_last_3s`: coefficient `-0.001747` (lowers CT win probability)
- `lag_02__T5__duck_amount`: coefficient `0.001708` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001708` (raises CT win probability)
- `lag_00__CT1__hp`: coefficient `0.001684` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `162340`, seconds `66.50`, LSTM delta `-0.3457`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009808`
- `lag_00__T_shots_fired_sum`: contribution `-0.009318`
- `lag_02__T_shots_fired_sum`: contribution `-0.008665`
- `lag_04__T_place_MAINHALL`: contribution `-0.008117`
- `lag_02__T5__shots_fired`: contribution `-0.007163`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `162244`, seconds `65.00`, LSTM delta `-0.1487`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009808`
- `lag_00__T_shots_fired_sum`: contribution `-0.009318`
- `lag_14__T5__duck_amount`: contribution `-0.006076`
- `lag_05__T_place_MAINHALL`: contribution `-0.005971`
- `lag_00__kill_diff_last_3s`: contribution `-0.005946`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `162276`, seconds `65.50`, LSTM delta `+0.1295`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.018636`
- `lag_02__T5__duck_amount`: contribution `+0.006275`
- `lag_14__T5__duck_amount`: contribution `+0.006076`
- `lag_00__kill_diff_last_3s`: contribution `+0.005946`
- `lag_03__CT3__duck_amount`: contribution `+0.005559`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `159908`, seconds `28.50`, LSTM delta `+0.1031`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.072215`
- `lag_00__T3__shots_fired`: contribution `+0.011440`
- `lag_02__T_shots_fired_sum`: contribution `+0.010831`
- `lag_14__T5__duck_amount`: contribution `+0.006076`
- `lag_03__T_kills_last_3s`: contribution `-0.004471`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.002226`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001454`

### tick `159812`, seconds `27.00`, LSTM delta `-0.1009`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.020966`
- `lag_00__T_kills_last_3s`: contribution `-0.009808`
- `lag_08__T_shots_fired_sum`: contribution `-0.006674`
- `lag_00__kill_diff_last_3s`: contribution `-0.005946`
- `lag_08__T5__shots_fired`: contribution `-0.003682`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.001224`
