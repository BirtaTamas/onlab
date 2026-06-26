# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `105506`, seconds `61.00`, LSTM `0.5070`, delta `+0.3913`
- tick `105058`, seconds `54.00`, LSTM `0.5211`, delta `+0.2379`
- tick `103042`, seconds `22.50`, LSTM `0.7292`, delta `+0.2299`
- tick `105314`, seconds `58.00`, LSTM `0.3819`, delta `-0.2084`
- tick `103362`, seconds `27.50`, LSTM `0.5091`, delta `-0.1555`
- tick `105346`, seconds `58.50`, LSTM `0.2377`, delta `-0.1442`
- tick `103426`, seconds `28.50`, LSTM `0.3149`, delta `-0.1172`
- tick `105698`, seconds `64.00`, LSTM `0.8582`, delta `+0.1084`
- tick `105538`, seconds `61.50`, LSTM `0.6092`, delta `+0.1022`
- tick `105602`, seconds `62.50`, LSTM `0.7097`, delta `+0.0827`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004888`, |coef| `0.004888`
- `lag_09__CT_place_TRAMP`: coefficient `-0.004806`, |coef| `0.004806`
- `lag_00__kill_diff_last_3s`: coefficient `0.004466`, |coef| `0.004466`
- `lag_13__CT_place_STAIRS`: coefficient `-0.003265`, |coef| `0.003265`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.003205`, |coef| `0.003205`
- `lag_00__damage_diff_last_5s`: coefficient `0.003000`, |coef| `0.003000`
- `lag_00__CT_kills_last_3s`: coefficient `0.002991`, |coef| `0.002991`
- `lag_15__CT_place_STAIRS`: coefficient `0.002884`, |coef| `0.002884`
- `lag_01__CT_defusing_count`: coefficient `0.002655`, |coef| `0.002655`
- `lag_00__T_kills_last_3s`: coefficient `-0.002596`, |coef| `0.002596`
- `lag_08__CT_place_TRAMP`: coefficient `-0.002590`, |coef| `0.002590`
- `lag_02__T_place_CTSPAWN`: coefficient `-0.002405`, |coef| `0.002405`
- `lag_01__CT_place_STAIRS`: coefficient `0.002374`, |coef| `0.002374`
- `lag_01__kill_diff_last_3s`: coefficient `0.002307`, |coef| `0.002307`
- `lag_07__CT_place_STAIRS`: coefficient `0.002292`, |coef| `0.002292`

## Top 10 utility ridge features

- `lag_10__CT4__smoke`: coefficient `-0.001118` (lowers CT win probability)
- `lag_05__T_A_site_active_smokes`: coefficient `-0.000858` (lowers CT win probability)
- `lag_04__CT4__smoke`: coefficient `0.000666` (raises CT win probability)
- `lag_05__T_active_smokes`: coefficient `-0.000629` (lowers CT win probability)
- `lag_11__CT4__smoke`: coefficient `-0.000626` (lowers CT win probability)
- `lag_05__CT4__smoke`: coefficient `0.000607` (raises CT win probability)
- `lag_06__T_A_site_active_smokes`: coefficient `-0.000521` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000474` (lowers CT win probability)
- `lag_10__CT4__utility_total`: coefficient `-0.000470` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.000444` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004888` (raises CT win probability)
- `lag_09__CT_place_TRAMP`: coefficient `-0.004806` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004466` (raises CT win probability)
- `lag_13__CT_place_STAIRS`: coefficient `-0.003265` (lowers CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.003205` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003000` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002991` (raises CT win probability)
- `lag_15__CT_place_STAIRS`: coefficient `0.002884` (raises CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.002655` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002596` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `105506`, seconds `61.00`, LSTM delta `+0.3913`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.047382`
- `lag_13__CT_place_STAIRS`: contribution `+0.025412`
- `lag_15__CT_place_STAIRS`: contribution `+0.022450`
- `lag_00__kill_diff_last_3s`: contribution `+0.021498`
- `lag_00__T_place_CTSPAWN`: contribution `+0.015288`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105058`, seconds `54.00`, LSTM delta `+0.2379`

Top all feature movements:
- `lag_09__CT_place_TRAMP`: contribution `+0.064753`
- `lag_01__CT_place_STAIRS`: contribution `+0.018476`
- `lag_00__kill_diff_last_3s`: contribution `+0.010749`
- `lag_00__CT_kills_last_3s`: contribution `+0.008636`
- `lag_03__CT_place_PALACEINTERIOR`: contribution `+0.007500`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103042`, seconds `22.50`, LSTM delta `+0.2299`

Top all feature movements:
- `lag_00__T_place_SCAFFOLDING`: contribution `+0.111769`
- `lag_03__T_place_SCAFFOLDING`: contribution `+0.036244`
- `lag_00__kill_diff_last_3s`: contribution `+0.010749`
- `lag_00__CT_kills_last_3s`: contribution `+0.008636`
- `lag_00__damage_diff_last_5s`: contribution `+0.006767`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105314`, seconds `58.00`, LSTM delta `-0.2084`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `-0.017836`
- `lag_02__T_place_CTSPAWN`: contribution `-0.011472`
- `lag_00__kill_diff_last_3s`: contribution `-0.010749`
- `lag_07__CT_place_JUNGLE`: contribution `-0.009280`
- `lag_00__T_kills_last_3s`: contribution `-0.008224`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103362`, seconds `27.50`, LSTM delta `-0.1555`

Top all feature movements:
- `lag_10__T_place_SCAFFOLDING`: contribution `-0.074021`
- `lag_00__damage_diff_last_5s`: contribution `-0.013535`
- `lag_00__kill_diff_last_3s`: contribution `-0.010749`
- `lag_01__CT_place_SIDEALLEY`: contribution `-0.010543`
- `lag_11__T_place_SCAFFOLDING`: contribution `+0.010199`

Top utility-only movements:
- No utility movement among the top local contributors.
