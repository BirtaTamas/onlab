# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `106727`, seconds `26.00`, LSTM `0.7386`, delta `+0.1938`
- tick `105863`, seconds `12.50`, LSTM `0.2306`, delta `-0.1933`
- tick `106471`, seconds `22.00`, LSTM `0.5645`, delta `+0.1520`
- tick `108135`, seconds `48.00`, LSTM `0.7796`, delta `-0.1469`
- tick `108167`, seconds `48.50`, LSTM `0.9204`, delta `+0.1407`
- tick `106791`, seconds `27.00`, LSTM `0.9182`, delta `+0.1160`
- tick `105831`, seconds `12.00`, LSTM `0.4239`, delta `-0.0824`
- tick `106151`, seconds `17.00`, LSTM `0.2849`, delta `+0.0684`
- tick `106759`, seconds `26.50`, LSTM `0.8022`, delta `+0.0636`
- tick `105895`, seconds `13.00`, LSTM `0.1756`, delta `-0.0550`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003325`, |coef| `0.003325`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.002853`, |coef| `0.002853`
- `lag_00__T_kills_last_3s`: coefficient `-0.002223`, |coef| `0.002223`
- `lag_00__CT_place_BANANA`: coefficient `0.002115`, |coef| `0.002115`
- `lag_12__CT_place_QUAD`: coefficient `0.002092`, |coef| `0.002092`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001989`, |coef| `0.001989`
- `lag_00__CT_kills_last_3s`: coefficient `0.001963`, |coef| `0.001963`
- `lag_00__damage_diff_last_5s`: coefficient `0.001831`, |coef| `0.001831`
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001791`, |coef| `0.001791`
- `lag_09__CT_place_TRAMP`: coefficient `-0.001773`, |coef| `0.001773`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001767`, |coef| `0.001767`
- `lag_15__T_place_LOWERMID`: coefficient `-0.001720`, |coef| `0.001720`
- `lag_11__T_place_BALCONY`: coefficient `-0.001715`, |coef| `0.001715`
- `lag_13__CT_place_ARCH`: coefficient `0.001670`, |coef| `0.001670`
- `lag_02__T2__is_walking`: coefficient `-0.001669`, |coef| `0.001669`

## Top 10 utility ridge features

- `lag_01__T_utility_damage_last_5s`: coefficient `-0.002853` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001989` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001791` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.001448` (lowers CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001381` (lowers CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.001335` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `0.001314` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.001292` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001258` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.001197` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003325` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002223` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.002115` (raises CT win probability)
- `lag_12__CT_place_QUAD`: coefficient `0.002092` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001963` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001831` (raises CT win probability)
- `lag_09__CT_place_TRAMP`: coefficient `-0.001773` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001767` (lowers CT win probability)
- `lag_15__T_place_LOWERMID`: coefficient `-0.001720` (lowers CT win probability)
- `lag_11__T_place_BALCONY`: coefficient `-0.001715` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `106727`, seconds `26.00`, LSTM delta `+0.1938`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `+0.023578`
- `lag_12__CT_place_QUAD`: contribution `+0.016486`
- `lag_01__CT_place_TOPOFMID`: contribution `+0.011857`
- `lag_12__T_place_BALCONY`: contribution `+0.008904`
- `lag_00__kill_diff_last_3s`: contribution `+0.008004`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `+0.005247`
- `lag_02__T2__flash_duration`: contribution `+0.005006`
- `lag_09__CT_active_infernos`: contribution `+0.002523`

### tick `105863`, seconds `12.50`, LSTM delta `-0.1933`

Top all feature movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.029324`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.011644`
- `lag_14__T_place_LOWERMID`: contribution `-0.009934`
- `lag_00__kill_diff_last_3s`: contribution `-0.008004`
- `lag_00__T_kills_last_3s`: contribution `-0.007042`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.029324`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.011644`
- `lag_01__T5__flash_duration`: contribution `-0.006306`

### tick `106471`, seconds `22.00`, LSTM delta `+0.1520`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `+0.019450`
- `lag_03__T_place_BALCONY`: contribution `+0.019190`
- `lag_12__CT_place_QUAD`: contribution `+0.016486`
- `lag_00__kill_diff_last_3s`: contribution `+0.008004`
- `lag_00__CT_kills_last_3s`: contribution `+0.005668`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `+0.003310`
- `lag_12__T5__flash_duration`: contribution `+0.003058`

### tick `108135`, seconds `48.00`, LSTM delta `-0.1469`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008004`
- `lag_00__T_kills_last_3s`: contribution `-0.007042`
- `lag_13__CT_place_ARCH`: contribution `-0.006815`
- `lag_00__CT_place_BANANA`: contribution `-0.006260`
- `lag_10__CT_place_ARCH`: contribution `-0.005709`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.004712`
- `lag_11__CT2__molly`: contribution `-0.002764`

### tick `108167`, seconds `48.50`, LSTM delta `+0.1407`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008004`
- `lag_00__CT_kills_last_3s`: contribution `+0.005668`
- `lag_11__CT_place_ARCH`: contribution `+0.005623`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.004224`
- `lag_02__CT_walking_count`: contribution `+0.004066`

Top utility-only movements:
- `lag_10__CT_A_site_active_infernos`: contribution `+0.004224`
