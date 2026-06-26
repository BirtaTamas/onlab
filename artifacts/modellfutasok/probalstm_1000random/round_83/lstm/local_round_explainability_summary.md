# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `78309`, seconds `57.50`, LSTM `0.2055`, delta `-0.3186`
- tick `78341`, seconds `58.00`, LSTM `0.0292`, delta `-0.1763`
- tick `78181`, seconds `55.50`, LSTM `0.5525`, delta `+0.1488`
- tick `78149`, seconds `55.00`, LSTM `0.4037`, delta `+0.0670`
- tick `75109`, seconds `7.50`, LSTM `0.2550`, delta `-0.0600`
- tick `75845`, seconds `19.00`, LSTM `0.2942`, delta `+0.0369`
- tick `75237`, seconds `9.50`, LSTM `0.2365`, delta `+0.0362`
- tick `75973`, seconds `21.00`, LSTM `0.2567`, delta `-0.0353`
- tick `76741`, seconds `33.00`, LSTM `0.3934`, delta `+0.0346`
- tick `75653`, seconds `16.00`, LSTM `0.2504`, delta `-0.0325`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001723`, |coef| `0.001723`
- `lag_00__T_kills_last_3s`: coefficient `-0.001611`, |coef| `0.001611`
- `lag_10__T1__flash_duration`: coefficient `-0.001566`, |coef| `0.001566`
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.001531`, |coef| `0.001531`
- `lag_00__CT1__flash_duration`: coefficient `0.001516`, |coef| `0.001516`
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `0.001443`, |coef| `0.001443`
- `lag_03__T2__flash_duration`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_00__CT3__flash_duration`: coefficient `0.001414`, |coef| `0.001414`
- `lag_02__T_place_TUNNELSTAIRS`: coefficient `0.001397`, |coef| `0.001397`
- `lag_05__T2__shots_fired`: coefficient `-0.001354`, |coef| `0.001354`
- `lag_00__CT_place_LONGDOORS`: coefficient `0.001314`, |coef| `0.001314`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001300`, |coef| `0.001300`
- `lag_02__T_place_LOWERTUNNEL`: coefficient `-0.001261`, |coef| `0.001261`
- `lag_10__T_flashed_players`: coefficient `-0.001214`, |coef| `0.001214`
- `lag_05__CT1__is_scoped`: coefficient `-0.001205`, |coef| `0.001205`

## Top 10 utility ridge features

- `lag_10__T1__flash_duration`: coefficient `-0.001566` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001516` (raises CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001436` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001414` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001300` (raises CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `-0.001194` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.001086` (lowers CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001037` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.001027` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000983` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001723` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001611` (lowers CT win probability)
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.001531` (lowers CT win probability)
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `0.001443` (raises CT win probability)
- `lag_02__T_place_TUNNELSTAIRS`: coefficient `0.001397` (raises CT win probability)
- `lag_05__T2__shots_fired`: coefficient `-0.001354` (lowers CT win probability)
- `lag_00__CT_place_LONGDOORS`: coefficient `0.001314` (raises CT win probability)
- `lag_02__T_place_LOWERTUNNEL`: coefficient `-0.001261` (lowers CT win probability)
- `lag_10__T_flashed_players`: coefficient `-0.001214` (lowers CT win probability)
- `lag_05__CT1__is_scoped`: coefficient `-0.001205` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78309`, seconds `57.50`, LSTM delta `-0.3186`

Top all feature movements:
- `lag_02__T_place_TUNNELSTAIRS`: contribution `-0.009753`
- `lag_00__CT3__flash_duration`: contribution `-0.009638`
- `lag_10__T1__flash_duration`: contribution `-0.008278`
- `lag_01__T_place_EXTENDEDA`: contribution `-0.007589`
- `lag_08__T_place_TUNNELSTAIRS`: contribution `-0.007242`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.009638`
- `lag_10__T1__flash_duration`: contribution `-0.008278`
- `lag_00__CT1__flash_duration`: contribution `-0.006736`
- `lag_00__CT_flash_duration_sum`: contribution `-0.006559`
- `lag_08__CT1__flash_duration`: contribution `-0.005309`

### tick `78341`, seconds `58.00`, LSTM delta `-0.1763`

Top all feature movements:
- `lag_03__T_place_TUNNELSTAIRS`: contribution `-0.010076`
- `lag_01__CT3__flash_duration`: contribution `-0.005848`
- `lag_11__T1__flash_duration`: contribution `-0.005741`
- `lag_02__T_place_EXTENDEDA`: contribution `-0.005158`
- `lag_00__T_kills_last_3s`: contribution `-0.005105`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `-0.005848`
- `lag_11__T1__flash_duration`: contribution `-0.005741`
- `lag_01__CT1__flash_duration`: contribution `-0.004567`
- `lag_01__CT_flash_duration_sum`: contribution `-0.004269`
- `lag_09__CT1__flash_duration`: contribution `-0.003705`

### tick `78181`, seconds `55.50`, LSTM delta `+0.1488`

Top all feature movements:
- `lag_04__T_place_TUNNELSTAIRS`: contribution `+0.006661`
- `lag_06__T1__flash_duration`: contribution `+0.005481`
- `lag_05__CT1__is_scoped`: contribution `+0.005159`
- `lag_10__CT1__is_scoped`: contribution `+0.004242`
- `lag_00__kill_diff_last_3s`: contribution `+0.004146`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.005481`
- `lag_04__CT1__flash_duration`: contribution `+0.003688`
- `lag_03__T2__flash_duration`: contribution `+0.002256`

### tick `78149`, seconds `55.00`, LSTM delta `+0.0670`

Top all feature movements:
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.010076`
- `lag_05__T1__flash_duration`: contribution `+0.005143`
- `lag_04__CT1__is_scoped`: contribution `+0.004733`
- `lag_09__CT1__is_scoped`: contribution `+0.004525`
- `lag_12__CT1__is_scoped`: contribution `+0.004431`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.005143`
- `lag_03__CT1__flash_duration`: contribution `+0.003077`
- `lag_03__CT3__flash_duration`: contribution `-0.001983`
- `lag_05__T_flash_duration_sum`: contribution `+0.001595`

### tick `75109`, seconds `7.50`, LSTM delta `-0.0600`

Top all feature movements:
- `lag_00__CT3__flash_duration`: contribution `+0.008149`
- `lag_00__CT_flashed_players`: contribution `+0.005101`
- `lag_00__CT_flash_duration_sum`: contribution `+0.003901`
- `lag_07__T_place_OUTSIDETUNNEL`: contribution `-0.003571`
- `lag_04__T_place_OUTSIDETUNNEL`: contribution `-0.003200`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `+0.008149`
- `lag_00__CT_flash_duration_sum`: contribution `+0.003901`
- `lag_00__CT3__molly`: contribution `-0.001225`
- `lag_15__T3__flash`: contribution `-0.001006`
- `lag_00__CT3__utility_total`: contribution `-0.000938`
