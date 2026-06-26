# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `15`

## Largest probability jumps

- tick `126614`, seconds `54.50`, LSTM `0.2508`, delta `-0.2661`
- tick `126806`, seconds `57.50`, LSTM `0.0360`, delta `-0.1394`
- tick `126646`, seconds `55.00`, LSTM `0.1553`, delta `-0.0955`
- tick `125846`, seconds `42.50`, LSTM `0.5304`, delta `-0.0490`
- tick `123190`, seconds `1.00`, LSTM `0.6097`, delta `+0.0348`
- tick `126774`, seconds `57.00`, LSTM `0.1755`, delta `+0.0325`
- tick `125302`, seconds `34.00`, LSTM `0.6060`, delta `+0.0245`
- tick `126678`, seconds `55.50`, LSTM `0.1312`, delta `-0.0241`
- tick `123542`, seconds `6.50`, LSTM `0.6255`, delta `+0.0215`
- tick `125590`, seconds `38.50`, LSTM `0.5739`, delta `-0.0208`

## Top 15 local ridge features

- `lag_11__T_place_TSTAIRS`: coefficient `0.003905`, |coef| `0.003905`
- `lag_00__T_kills_last_3s`: coefficient `-0.003012`, |coef| `0.003012`
- `lag_08__T_place_TMAIN`: coefficient `-0.002987`, |coef| `0.002987`
- `lag_05__CT_A_site_active_smokes`: coefficient `0.002973`, |coef| `0.002973`
- `lag_00__T_damage_last_5s`: coefficient `-0.002715`, |coef| `0.002715`
- `lag_00__damage_diff_last_5s`: coefficient `0.002614`, |coef| `0.002614`
- `lag_00__CT2__alive`: coefficient `0.002572`, |coef| `0.002572`
- `lag_00__CT2__hp`: coefficient `0.002543`, |coef| `0.002543`
- `lag_15__CT_place_CONNECTOR`: coefficient `0.002465`, |coef| `0.002465`
- `lag_00__CT2__armor`: coefficient `0.002408`, |coef| `0.002408`
- `lag_00__CT2__has_defuser`: coefficient `0.002309`, |coef| `0.002309`
- `lag_00__kill_diff_last_3s`: coefficient `0.002279`, |coef| `0.002279`
- `lag_00__CT_place_TUNNELS`: coefficient `0.002246`, |coef| `0.002246`
- `lag_05__CT_active_smokes`: coefficient `0.002185`, |coef| `0.002185`
- `lag_00__CT2__has_helmet`: coefficient `0.002136`, |coef| `0.002136`

## Top 10 utility ridge features

- `lag_05__CT_A_site_active_smokes`: coefficient `0.002973` (raises CT win probability)
- `lag_05__CT_active_smokes`: coefficient `0.002185` (raises CT win probability)
- `lag_04__CT3__flash`: coefficient `0.001841` (raises CT win probability)
- `lag_06__CT_A_site_active_smokes`: coefficient `0.001740` (raises CT win probability)
- `lag_05__CT_B_site_active_smokes`: coefficient `0.001579` (raises CT win probability)
- `lag_06__CT_active_smokes`: coefficient `0.001298` (raises CT win probability)
- `lag_05__active_smokes_total`: coefficient `0.001260` (raises CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `0.001155` (raises CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `0.001098` (raises CT win probability)
- `lag_05__CT3__flash`: coefficient `0.001079` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_TSTAIRS`: coefficient `0.003905` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003012` (lowers CT win probability)
- `lag_08__T_place_TMAIN`: coefficient `-0.002987` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002715` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002614` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.002572` (raises CT win probability)
- `lag_00__CT2__hp`: coefficient `0.002543` (raises CT win probability)
- `lag_15__CT_place_CONNECTOR`: coefficient `0.002465` (raises CT win probability)
- `lag_00__CT2__armor`: coefficient `0.002408` (raises CT win probability)
- `lag_00__CT2__has_defuser`: coefficient `0.002309` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `126614`, seconds `54.50`, LSTM delta `-0.2661`

Top all feature movements:
- `lag_11__T_place_TSTAIRS`: contribution `-0.022138`
- `lag_08__T_place_TMAIN`: contribution `-0.011584`
- `lag_05__CT_A_site_active_smokes`: contribution `-0.009570`
- `lag_00__T_kills_last_3s`: contribution `-0.009542`
- `lag_15__CT_place_CONNECTOR`: contribution `-0.008814`

Top utility-only movements:
- `lag_05__CT_A_site_active_smokes`: contribution `-0.009570`
- `lag_05__CT_active_smokes`: contribution `-0.005047`

### tick `126806`, seconds `57.50`, LSTM delta `-0.1394`

Top all feature movements:
- `lag_00__CT_place_TUNNELS`: contribution `-0.006874`
- `lag_00__T_damage_last_5s`: contribution `-0.006510`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005763`
- `lag_00__damage_diff_last_5s`: contribution `-0.005543`
- `lag_14__T1__duck_amount`: contribution `-0.003903`

Top utility-only movements:
- `lag_11__CT_A_site_active_smokes`: contribution `-0.002373`

### tick `126646`, seconds `55.00`, LSTM delta `-0.0955`

Top all feature movements:
- `lag_12__T_place_TSTAIRS`: contribution `-0.010799`
- `lag_06__CT_A_site_active_smokes`: contribution `-0.005602`
- `lag_09__T_place_TMAIN`: contribution `-0.005453`
- `lag_01__T_kills_last_3s`: contribution `-0.004552`
- `lag_00__CT3__is_walking`: contribution `+0.003738`

Top utility-only movements:
- `lag_06__CT_A_site_active_smokes`: contribution `-0.005602`
- `lag_06__CT_active_smokes`: contribution `-0.002998`

### tick `125846`, seconds `42.50`, LSTM delta `-0.0490`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009542`
- `lag_00__CT_place_TUNNELS`: contribution `-0.006874`
- `lag_00__T_damage_last_5s`: contribution `-0.006510`
- `lag_00__kill_diff_last_3s`: contribution `-0.005485`
- `lag_10__CT5__is_walking`: contribution `+0.003810`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `123190`, seconds `1.00`, LSTM delta `+0.0348`

Top all feature movements:
- `lag_00__T_he_last_5s`: contribution `+0.009013`
- `lag_01__CT_velocity_mean`: contribution `-0.002046`
- `lag_02__T_place_TSPAWN`: contribution `+0.001850`
- `lag_02__CT_place_CTSPAWN`: contribution `+0.001804`
- `lag_02__CT_closest_enemy_dist`: contribution `+0.001766`

Top utility-only movements:
- `lag_00__T_he_last_5s`: contribution `+0.009013`
- `lag_02__CT3__flash`: contribution `+0.000594`
- `lag_02__CT_utility_inv`: contribution `+0.000553`
- `lag_02__CT3__molly`: contribution `-0.000521`
- `lag_02__CT_flash_inv`: contribution `+0.000516`
