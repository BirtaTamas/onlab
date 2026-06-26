# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `25471`, seconds `62.00`, LSTM `0.5697`, delta `+0.3214`
- tick `25695`, seconds `65.50`, LSTM `0.8065`, delta `+0.2240`
- tick `24383`, seconds `45.00`, LSTM `0.3517`, delta `-0.1492`
- tick `25919`, seconds `69.00`, LSTM `0.9066`, delta `+0.1431`
- tick `26943`, seconds `85.00`, LSTM `0.9283`, delta `+0.1064`
- tick `26591`, seconds `79.50`, LSTM `0.8273`, delta `-0.1035`
- tick `24415`, seconds `45.50`, LSTM `0.3011`, delta `-0.0506`
- tick `24511`, seconds `47.00`, LSTM `0.3180`, delta `+0.0425`
- tick `24671`, seconds `49.50`, LSTM `0.2562`, delta `-0.0417`
- tick `23007`, seconds `23.50`, LSTM `0.5370`, delta `-0.0365`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003954`, |coef| `0.003954`
- `lag_08__CT1__duck_amount`: coefficient `0.003493`, |coef| `0.003493`
- `lag_00__CT_kills_last_3s`: coefficient `0.003459`, |coef| `0.003459`
- `lag_00__damage_diff_last_5s`: coefficient `0.003231`, |coef| `0.003231`
- `lag_08__CT1__is_walking`: coefficient `-0.002596`, |coef| `0.002596`
- `lag_00__CT_B_site_active_infernos`: coefficient `0.002548`, |coef| `0.002548`
- `lag_00__T_place_APARTMENTS`: coefficient `-0.002535`, |coef| `0.002535`
- `lag_07__T1__duck_amount`: coefficient `0.002481`, |coef| `0.002481`
- `lag_06__T_place_UNDERPASS`: coefficient `-0.002467`, |coef| `0.002467`
- `lag_00__T4__flash`: coefficient `-0.002318`, |coef| `0.002318`
- `lag_00__T5__duck_amount`: coefficient `0.002281`, |coef| `0.002281`
- `lag_00__CT_damage_last_5s`: coefficient `0.002245`, |coef| `0.002245`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002105`, |coef| `0.002105`
- `lag_00__T4__alive`: coefficient `-0.002097`, |coef| `0.002097`
- `lag_10__T_flashed_players`: coefficient `0.002094`, |coef| `0.002094`

## Top 10 utility ridge features

- `lag_00__CT_B_site_active_infernos`: coefficient `0.002548` (raises CT win probability)
- `lag_00__T4__flash`: coefficient `-0.002318` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001990` (lowers CT win probability)
- `lag_00__active_infernos_total`: coefficient `0.001886` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001860` (lowers CT win probability)
- `lag_03__CT1__molly`: coefficient `-0.001814` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.001772` (raises CT win probability)
- `lag_03__T2__molly`: coefficient `-0.001650` (lowers CT win probability)
- `lag_12__T_active_smokes`: coefficient `0.001645` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.001581` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003954` (raises CT win probability)
- `lag_08__CT1__duck_amount`: coefficient `0.003493` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003459` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003231` (raises CT win probability)
- `lag_08__CT1__is_walking`: coefficient `-0.002596` (lowers CT win probability)
- `lag_00__T_place_APARTMENTS`: coefficient `-0.002535` (lowers CT win probability)
- `lag_07__T1__duck_amount`: coefficient `0.002481` (raises CT win probability)
- `lag_06__T_place_UNDERPASS`: coefficient `-0.002467` (lowers CT win probability)
- `lag_00__T5__duck_amount`: coefficient `0.002281` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002245` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25471`, seconds `62.00`, LSTM delta `+0.3214`

Top all feature movements:
- `lag_08__CT1__duck_amount`: contribution `+0.013327`
- `lag_00__CT_kills_last_3s`: contribution `+0.009987`
- `lag_07__T1__duck_amount`: contribution `+0.009716`
- `lag_06__T_place_UNDERPASS`: contribution `+0.009665`
- `lag_00__kill_diff_last_3s`: contribution `+0.009518`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `+0.008754`
- `lag_00__T4__flash`: contribution `+0.006299`
- `lag_00__active_infernos_total`: contribution `+0.005420`
- `lag_00__T_B_site_active_infernos`: contribution `+0.005009`

### tick `25695`, seconds `65.50`, LSTM delta `+0.2240`

Top all feature movements:
- `lag_08__CT1__duck_amount`: contribution `+0.013327`
- `lag_00__CT_kills_last_3s`: contribution `+0.009987`
- `lag_00__kill_diff_last_3s`: contribution `+0.009518`
- `lag_00__damage_diff_last_5s`: contribution `+0.007290`
- `lag_05__T_place_CTSPAWN`: contribution `+0.006822`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `+0.003640`

### tick `24383`, seconds `45.00`, LSTM delta `-0.1492`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.011011`
- `lag_00__kill_diff_last_3s`: contribution `-0.009518`
- `lag_02__CT_place_UNDERPASS`: contribution `-0.006904`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.006465`
- `lag_06__CT_place_TRUCK`: contribution `-0.005398`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25919`, seconds `69.00`, LSTM delta `+0.1431`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009987`
- `lag_00__kill_diff_last_3s`: contribution `+0.009518`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007313`
- `lag_00__damage_diff_last_5s`: contribution `+0.007290`
- `lag_02__CT_place_UNDERPASS`: contribution `+0.006904`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `-0.005009`
- `lag_00__T_active_infernos`: contribution `-0.002718`
- `lag_00__active_infernos_total`: contribution `-0.002710`

### tick `26943`, seconds `85.00`, LSTM delta `+0.1064`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009987`
- `lag_00__kill_diff_last_3s`: contribution `+0.009518`
- `lag_08__CT1__duck_amount`: contribution `+0.006576`
- `lag_00__T_place_APARTMENTS`: contribution `+0.005803`
- `lag_00__CT3__is_scoped`: contribution `+0.005653`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `-0.001824`
