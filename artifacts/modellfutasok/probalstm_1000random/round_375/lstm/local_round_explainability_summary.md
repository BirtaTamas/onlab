# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `92247`, seconds `65.00`, LSTM `0.8378`, delta `+0.2443`
- tick `88791`, seconds `11.00`, LSTM `0.1718`, delta `-0.1473`
- tick `91927`, seconds `60.00`, LSTM `0.1673`, delta `+0.1373`
- tick `92183`, seconds `64.00`, LSTM `0.5586`, delta `+0.1324`
- tick `89271`, seconds `18.50`, LSTM `0.0918`, delta `-0.0776`
- tick `91991`, seconds `61.00`, LSTM `0.2933`, delta `+0.0680`
- tick `92023`, seconds `61.50`, LSTM `0.3574`, delta `+0.0642`
- tick `89239`, seconds `18.00`, LSTM `0.1695`, delta `+0.0620`
- tick `88599`, seconds `8.00`, LSTM `0.2124`, delta `-0.0591`
- tick `91959`, seconds `60.50`, LSTM `0.2253`, delta `+0.0579`

## Top 15 local ridge features

- `lag_13__T_place_HOLE`: coefficient `0.002936`, |coef| `0.002936`
- `lag_11__T_place_HOLE`: coefficient `0.002186`, |coef| `0.002186`
- `lag_06__CT_place_OUTSIDELONG`: coefficient `0.001942`, |coef| `0.001942`
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.001877`, |coef| `0.001877`
- `lag_00__CT_place_HOLE`: coefficient `0.001866`, |coef| `0.001866`
- `lag_00__T_place_BDOORS`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_00__T5__is_scoped`: coefficient `0.001772`, |coef| `0.001772`
- `lag_00__kill_diff_last_3s`: coefficient `0.001735`, |coef| `0.001735`
- `lag_07__CT_place_OUTSIDELONG`: coefficient `0.001701`, |coef| `0.001701`
- `lag_10__T_place_HOLE`: coefficient `-0.001681`, |coef| `0.001681`
- `lag_00__T_place_HOLE`: coefficient `-0.001570`, |coef| `0.001570`
- `lag_12__T_place_HOLE`: coefficient `0.001498`, |coef| `0.001498`
- `lag_08__CT_place_OUTSIDELONG`: coefficient `0.001422`, |coef| `0.001422`
- `lag_14__T_place_HOLE`: coefficient `0.001376`, |coef| `0.001376`
- `lag_06__T_place_HOLE`: coefficient `0.001360`, |coef| `0.001360`

## Top 10 utility ridge features

- `lag_06__CT1__flash_duration`: coefficient `-0.000809` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.000699` (raises CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.000677` (raises CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000650` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000650` (lowers CT win probability)
- `lag_05__active_infernos_total`: coefficient `-0.000630` (lowers CT win probability)
- `lag_12__T4__molly`: coefficient `-0.000615` (lowers CT win probability)
- `lag_07__T4__smoke`: coefficient `-0.000612` (lowers CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000597` (lowers CT win probability)
- `lag_02__T2__smoke`: coefficient `-0.000594` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_HOLE`: coefficient `0.002936` (raises CT win probability)
- `lag_11__T_place_HOLE`: coefficient `0.002186` (raises CT win probability)
- `lag_06__CT_place_OUTSIDELONG`: coefficient `0.001942` (raises CT win probability)
- `lag_09__CT_place_OUTSIDELONG`: coefficient `0.001877` (raises CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.001866` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.001779` (lowers CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.001772` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001735` (raises CT win probability)
- `lag_07__CT_place_OUTSIDELONG`: coefficient `0.001701` (raises CT win probability)
- `lag_10__T_place_HOLE`: coefficient `-0.001681` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `92247`, seconds `65.00`, LSTM delta `+0.2443`

Top all feature movements:
- `lag_13__T_place_HOLE`: contribution `+0.075675`
- `lag_10__T_place_HOLE`: contribution `+0.043321`
- `lag_06__CT_place_OUTSIDELONG`: contribution `+0.019696`
- `lag_01__CT_place_OUTSIDELONG`: contribution `+0.006273`
- `lag_06__CT_place_LONGDOORS`: contribution `+0.005434`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88791`, seconds `11.00`, LSTM delta `-0.1473`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `-0.008452`
- `lag_04__CT_place_SHORTSTAIRS`: contribution `-0.006096`
- `lag_04__CT_place_BDOORS`: contribution `-0.005589`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.004716`
- `lag_08__CT_place_EXTENDEDA`: contribution `-0.004610`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `-0.003695`

### tick `91927`, seconds `60.00`, LSTM delta `+0.1373`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.040470`
- `lag_03__T_place_HOLE`: contribution `+0.015419`
- `lag_03__T_place_BDOORS`: contribution `+0.015337`
- `lag_10__T_place_BDOORS`: contribution `+0.009698`
- `lag_13__CT3__is_scoped`: contribution `+0.004238`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.001439`
- `lag_00__T2__smoke`: contribution `+0.001429`

### tick `92183`, seconds `64.00`, LSTM delta `+0.1324`

Top all feature movements:
- `lag_11__T_place_HOLE`: contribution `+0.056359`
- `lag_08__T_place_HOLE`: contribution `+0.010018`
- `lag_04__CT_place_OUTSIDELONG`: contribution `+0.006954`
- `lag_04__CT_place_LONGDOORS`: contribution `+0.004841`
- `lag_00__kill_diff_last_3s`: contribution `+0.004176`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89271`, seconds `18.50`, LSTM delta `-0.0776`

Top all feature movements:
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.005380`
- `lag_06__CT3__is_scoped`: contribution `-0.005225`
- `lag_01__CT_place_CATWALK`: contribution `-0.004282`
- `lag_13__CT3__is_scoped`: contribution `-0.004238`
- `lag_11__CT_place_LONGDOORS`: contribution `-0.004043`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `-0.001539`
