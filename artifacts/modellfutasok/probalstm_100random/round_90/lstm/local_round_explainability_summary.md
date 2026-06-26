# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `11`

## Largest probability jumps

- tick `92731`, seconds `91.50`, LSTM `0.7412`, delta `+0.2043`
- tick `92795`, seconds `92.50`, LSTM `0.9388`, delta `+0.1996`
- tick `92635`, seconds `90.00`, LSTM `0.4843`, delta `-0.1735`
- tick `91899`, seconds `78.50`, LSTM `0.5203`, delta `-0.1697`
- tick `91867`, seconds `78.00`, LSTM `0.6900`, delta `+0.1496`
- tick `93499`, seconds `103.50`, LSTM `0.8025`, delta `-0.1397`
- tick `92443`, seconds `87.00`, LSTM `0.5851`, delta `+0.1155`
- tick `93531`, seconds `104.00`, LSTM `0.9169`, delta `+0.1144`
- tick `92699`, seconds `91.00`, LSTM `0.5369`, delta `+0.0839`
- tick `91803`, seconds `77.00`, LSTM `0.5492`, delta `-0.0482`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003162`, |coef| `0.003162`
- `lag_00__CT_kills_last_3s`: coefficient `0.002629`, |coef| `0.002629`
- `lag_06__T2__flash_duration`: coefficient `0.002021`, |coef| `0.002021`
- `lag_00__damage_diff_last_5s`: coefficient `0.001874`, |coef| `0.001874`
- `lag_08__T2__flash_duration`: coefficient `0.001701`, |coef| `0.001701`
- `lag_08__T_place_ELECTRICALBOX`: coefficient `0.001647`, |coef| `0.001647`
- `lag_11__CT1__flash_duration`: coefficient `-0.001385`, |coef| `0.001385`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.001322`, |coef| `0.001322`
- `lag_00__T_macro_A`: coefficient `-0.001322`, |coef| `0.001322`
- `lag_00__T_kills_last_3s`: coefficient `-0.001277`, |coef| `0.001277`
- `lag_00__CT_damage_last_5s`: coefficient `0.001261`, |coef| `0.001261`
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001244`, |coef| `0.001244`
- `lag_05__T_place_ELECTRICALBOX`: coefficient `0.001233`, |coef| `0.001233`
- `lag_04__T_shots_fired_sum`: coefficient `-0.001229`, |coef| `0.001229`
- `lag_07__T2__flash_duration`: coefficient `0.001218`, |coef| `0.001218`

## Top 10 utility ridge features

- `lag_06__T2__flash_duration`: coefficient `0.002021` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.001701` (raises CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.001385` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001244` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.001218` (raises CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001176` (lowers CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.001110` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001097` (raises CT win probability)
- `lag_11__CT_flash_duration_sum`: coefficient `-0.000930` (lowers CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000920` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003162` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002629` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001874` (raises CT win probability)
- `lag_08__T_place_ELECTRICALBOX`: coefficient `0.001647` (raises CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.001322` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.001322` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001277` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001261` (raises CT win probability)
- `lag_05__T_place_ELECTRICALBOX`: coefficient `0.001233` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `-0.001229` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `92731`, seconds `91.50`, LSTM delta `+0.2043`

Top all feature movements:
- `lag_06__T2__flash_duration`: contribution `+0.015064`
- `lag_00__kill_diff_last_3s`: contribution `+0.007612`
- `lag_00__CT_kills_last_3s`: contribution `+0.007591`
- `lag_09__T_place_LONGDOG`: contribution `+0.005271`
- `lag_13__CT1__is_scoped`: contribution `+0.004611`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `+0.015064`
- `lag_06__T_flash_duration_sum`: contribution `+0.003387`
- `lag_04__T_A_site_active_infernos`: contribution `+0.003305`

### tick `92795`, seconds `92.50`, LSTM delta `+0.1996`

Top all feature movements:
- `lag_08__T2__flash_duration`: contribution `+0.012680`
- `lag_00__kill_diff_last_3s`: contribution `+0.007612`
- `lag_00__CT_kills_last_3s`: contribution `+0.007591`
- `lag_05__kill_diff_last_3s`: contribution `+0.005084`
- `lag_03__T_A_site_active_infernos`: contribution `+0.003704`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `+0.012680`
- `lag_03__T_A_site_active_infernos`: contribution `+0.003704`
- `lag_08__T_flash_duration_sum`: contribution `+0.003058`

### tick `92635`, seconds `90.00`, LSTM delta `-0.1735`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.015223`
- `lag_03__T2__flash_duration`: contribution `-0.008770`
- `lag_00__CT_kills_last_3s`: contribution `-0.007591`
- `lag_09__T_place_LONGDOG`: contribution `-0.005271`
- `lag_03__T_flashed_players`: contribution `-0.004677`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `-0.008770`
- `lag_03__T_flash_duration_sum`: contribution `-0.002883`

### tick `91899`, seconds `78.50`, LSTM delta `-0.1697`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.015929`
- `lag_00__T2__shots_fired`: contribution `-0.013799`
- `lag_11__CT1__flash_duration`: contribution `-0.007645`
- `lag_00__kill_diff_last_3s`: contribution `-0.007612`
- `lag_13__T_place_DUMPSTER`: contribution `-0.005854`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `-0.007645`
- `lag_11__CT_flash_duration_sum`: contribution `-0.004246`
- `lag_03__CT4__flash_duration`: contribution `-0.004182`
- `lag_11__CT3__flash_duration`: contribution `-0.002241`

### tick `91867`, seconds `78.00`, LSTM delta `+0.1496`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.007612`
- `lag_00__CT_kills_last_3s`: contribution `+0.007591`
- `lag_12__T_place_DUMPSTER`: contribution `+0.006158`
- `lag_00__T_shots_fired_sum`: contribution `+0.005030`
- `lag_00__damage_diff_last_5s`: contribution `+0.004102`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.001988`
- `lag_10__CT3__flash_duration`: contribution `+0.001868`
