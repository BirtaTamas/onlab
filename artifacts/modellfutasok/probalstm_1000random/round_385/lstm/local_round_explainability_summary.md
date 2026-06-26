# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `76751`, seconds `51.50`, LSTM `0.5685`, delta `+0.3035`
- tick `74639`, seconds `18.50`, LSTM `0.2904`, delta `+0.2048`
- tick `77167`, seconds `58.00`, LSTM `0.6478`, delta `+0.2047`
- tick `75919`, seconds `38.50`, LSTM `0.1246`, delta `-0.1719`
- tick `77455`, seconds `62.50`, LSTM `0.8012`, delta `+0.1372`
- tick `76879`, seconds `53.50`, LSTM `0.5274`, delta `-0.1177`
- tick `77487`, seconds `63.00`, LSTM `0.8932`, delta `+0.0920`
- tick `76783`, seconds `52.00`, LSTM `0.6531`, delta `+0.0846`
- tick `76463`, seconds `47.00`, LSTM `0.2152`, delta `+0.0606`
- tick `73487`, seconds `0.50`, LSTM `0.0516`, delta `-0.0574`

## Top 15 local ridge features

- `lag_09__CT_place_LOWERMID`: coefficient `-0.004329`, |coef| `0.004329`
- `lag_00__CT_place_LOWERMID`: coefficient `-0.003710`, |coef| `0.003710`
- `lag_13__CT_place_TRAMP`: coefficient `-0.003551`, |coef| `0.003551`
- `lag_00__CT_place_UPSTAIRS`: coefficient `-0.003401`, |coef| `0.003401`
- `lag_00__CT_kills_last_3s`: coefficient `0.002871`, |coef| `0.002871`
- `lag_09__CT_place_TRAMP`: coefficient `0.002686`, |coef| `0.002686`
- `lag_00__kill_diff_last_3s`: coefficient `0.002497`, |coef| `0.002497`
- `lag_09__CT_place_UPSTAIRS`: coefficient `-0.002387`, |coef| `0.002387`
- `lag_00__damage_diff_last_5s`: coefficient `0.002152`, |coef| `0.002152`
- `lag_00__CT_damage_last_5s`: coefficient `0.002136`, |coef| `0.002136`
- `lag_00__CT_place_TRAMP`: coefficient `-0.002124`, |coef| `0.002124`
- `lag_06__CT3__flash_duration`: coefficient `0.001752`, |coef| `0.001752`
- `lag_05__CT_place_BALCONY`: coefficient `-0.001604`, |coef| `0.001604`
- `lag_10__CT_place_TRAMP`: coefficient `0.001560`, |coef| `0.001560`
- `lag_01__CT_place_UPSTAIRS`: coefficient `-0.001549`, |coef| `0.001549`

## Top 10 utility ridge features

- `lag_06__CT3__flash_duration`: coefficient `0.001752` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.001208` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.001156` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.001146` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.001121` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001109` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.001075` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001064` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001033` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.001006` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_LOWERMID`: coefficient `-0.004329` (lowers CT win probability)
- `lag_00__CT_place_LOWERMID`: coefficient `-0.003710` (lowers CT win probability)
- `lag_13__CT_place_TRAMP`: coefficient `-0.003551` (lowers CT win probability)
- `lag_00__CT_place_UPSTAIRS`: coefficient `-0.003401` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002871` (raises CT win probability)
- `lag_09__CT_place_TRAMP`: coefficient `0.002686` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002497` (raises CT win probability)
- `lag_09__CT_place_UPSTAIRS`: coefficient `-0.002387` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002152` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002136` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76751`, seconds `51.50`, LSTM delta `+0.3035`

Top all feature movements:
- `lag_09__CT_place_LOWERMID`: contribution `+0.118754`
- `lag_09__CT_place_TRAMP`: contribution `+0.036180`
- `lag_00__CT_place_TRAMP`: contribution `+0.028619`
- `lag_05__CT_place_BALCONY`: contribution `+0.010292`
- `lag_00__CT_kills_last_3s`: contribution `+0.008289`

Top utility-only movements:
- `lag_00__T5__flash`: contribution `+0.003052`
- `lag_00__T5__utility_total`: contribution `+0.002479`

### tick `74639`, seconds `18.50`, LSTM delta `+0.2048`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008289`
- `lag_04__T_shots_fired_sum`: contribution `+0.007615`
- `lag_13__CT_place_BALCONY`: contribution `+0.007333`
- `lag_00__kill_diff_last_3s`: contribution `+0.006010`
- `lag_04__T5__shots_fired`: contribution `+0.005259`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.003013`
- `lag_02__T_A_site_active_infernos`: contribution `+0.002420`

### tick `77167`, seconds `58.00`, LSTM delta `+0.2047`

Top all feature movements:
- `lag_13__CT_place_TRAMP`: contribution `+0.047837`
- `lag_00__CT_kills_last_3s`: contribution `+0.008289`
- `lag_06__CT_place_LIBRARY`: contribution `+0.006722`
- `lag_00__kill_diff_last_3s`: contribution `+0.006010`
- `lag_02__T3__flash_duration`: contribution `+0.005713`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.005713`
- `lag_00__T3__flash_duration`: contribution `+0.003939`
- `lag_02__T_B_site_active_infernos`: contribution `+0.002258`

### tick `75919`, seconds `38.50`, LSTM delta `-0.1719`

Top all feature movements:
- `lag_00__CT_place_UPSTAIRS`: contribution `-0.138855`
- `lag_07__CT_place_BRIDGE`: contribution `-0.008030`
- `lag_00__CT_place_BRIDGE`: contribution `-0.006983`
- `lag_03__T_place_ARCH`: contribution `-0.002717`
- `lag_09__CT_place_RUINS`: contribution `-0.002450`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77455`, seconds `62.50`, LSTM delta `+0.1372`

Top all feature movements:
- `lag_06__CT3__flash_duration`: contribution `+0.011120`
- `lag_15__CT_place_LIBRARY`: contribution `+0.006511`
- `lag_11__CT_place_LIBRARY`: contribution `+0.005969`
- `lag_11__T3__flash_duration`: contribution `+0.005901`
- `lag_06__CT_flashed_players`: contribution `+0.005256`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.011120`
- `lag_11__T3__flash_duration`: contribution `+0.005901`
- `lag_06__CT_flash_duration_sum`: contribution `+0.005034`
- `lag_09__T3__flash_duration`: contribution `+0.003946`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.003437`
