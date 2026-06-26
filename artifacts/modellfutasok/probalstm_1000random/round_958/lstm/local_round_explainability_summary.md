# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `82686`, seconds `31.00`, LSTM `0.1445`, delta `-0.2614`
- tick `81854`, seconds `18.00`, LSTM `0.2671`, delta `-0.2582`
- tick `82910`, seconds `34.50`, LSTM `0.0317`, delta `-0.1705`
- tick `82590`, seconds `29.50`, LSTM `0.4284`, delta `+0.1049`
- tick `82878`, seconds `34.00`, LSTM `0.2022`, delta `+0.0850`
- tick `81886`, seconds `18.50`, LSTM `0.2047`, delta `-0.0625`
- tick `81662`, seconds `15.00`, LSTM `0.5542`, delta `+0.0585`
- tick `82174`, seconds `23.00`, LSTM `0.2339`, delta `+0.0521`
- tick `81918`, seconds `19.00`, LSTM `0.1539`, delta `-0.0508`
- tick `82046`, seconds `21.00`, LSTM `0.1335`, delta `+0.0437`

## Top 15 local ridge features

- `lag_00__utility_damage_diff_last_5s`: coefficient `0.002823`, |coef| `0.002823`
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.002464`, |coef| `0.002464`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002220`, |coef| `0.002220`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.002187`, |coef| `0.002187`
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001769`, |coef| `0.001769`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001730`, |coef| `0.001730`
- `lag_00__T_kills_last_3s`: coefficient `-0.001606`, |coef| `0.001606`
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001503`, |coef| `0.001503`
- `lag_04__CT3__flash_duration`: coefficient `-0.001478`, |coef| `0.001478`
- `lag_08__T2__flash_duration`: coefficient `0.001441`, |coef| `0.001441`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001410`, |coef| `0.001410`
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.001303`, |coef| `0.001303`
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001288`, |coef| `0.001288`
- `lag_00__CT3__flash_duration`: coefficient `0.001215`, |coef| `0.001215`
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.001206`, |coef| `0.001206`

## Top 10 utility ridge features

- `lag_00__utility_damage_diff_last_5s`: coefficient `0.002823` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.002464` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002220` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.002187` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001769` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001730` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001503` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.001478` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.001441` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001288` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001606` (lowers CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001410` (raises CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.001303` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001145` (raises CT win probability)
- `lag_10__T_place_TRAMP`: coefficient `0.001131` (raises CT win probability)
- `lag_14__T_place_TRAMP`: coefficient `-0.001083` (lowers CT win probability)
- `lag_12__CT_place_APARTMENTS`: coefficient `-0.000999` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000993` (raises CT win probability)
- `lag_12__CT_place_BALCONY`: coefficient `0.000991` (raises CT win probability)
- `lag_06__CT5__is_walking`: coefficient `0.000980` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `82686`, seconds `31.00`, LSTM delta `-0.2614`

Top all feature movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.039594`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.013939`
- `lag_04__CT3__flash_duration`: contribution `-0.010037`
- `lag_08__T2__flash_duration`: contribution `-0.009379`
- `lag_00__CT3__flash_duration`: contribution `-0.008256`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.039594`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.013939`
- `lag_04__CT3__flash_duration`: contribution `-0.010037`
- `lag_08__T2__flash_duration`: contribution `-0.009379`
- `lag_00__CT3__flash_duration`: contribution `-0.008256`

### tick `81854`, seconds `18.00`, LSTM delta `-0.2582`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.016483`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.014990`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.013255`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.007664`
- `lag_10__T_place_TRAMP`: contribution `-0.006621`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.016483`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.014990`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.013255`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.007664`
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.005585`

### tick `82910`, seconds `34.50`, LSTM delta `-0.1705`

Top all feature movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.032882`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.027995`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013108`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.010278`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.005706`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.032882`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.027995`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.010278`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.005706`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.004222`

### tick `82590`, seconds `29.50`, LSTM delta `+0.1049`

Top all feature movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.032627`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.027805`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005706`
- `lag_01__CT3__flash_duration`: contribution `+0.004407`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.004062`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.032627`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.027805`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005706`
- `lag_01__CT3__flash_duration`: contribution `+0.004407`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.004062`

### tick `82878`, seconds `34.00`, LSTM delta `+0.0850`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007589`
- `lag_00__CT_place_TOPOFMID`: contribution `+0.005116`
- `lag_00__T_kills_last_3s`: contribution `+0.005088`
- `lag_00__CT_place_APARTMENTS`: contribution `+0.005007`
- `lag_10__CT3__flash_duration`: contribution `+0.004522`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `+0.004522`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.004121`
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.003632`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003314`
- `lag_12__T_B_site_active_infernos`: contribution `+0.002290`
