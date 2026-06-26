# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `132279`, seconds `106.00`, LSTM `0.7696`, delta `+0.2987`
- tick `133495`, seconds `125.00`, LSTM `0.9369`, delta `+0.2480`
- tick `129719`, seconds `66.00`, LSTM `0.5138`, delta `+0.2137`
- tick `130615`, seconds `80.00`, LSTM `0.1600`, delta `-0.1434`
- tick `133079`, seconds `118.50`, LSTM `0.7685`, delta `-0.1204`
- tick `131831`, seconds `99.00`, LSTM `0.5411`, delta `+0.1045`
- tick `133303`, seconds `122.00`, LSTM `0.7210`, delta `+0.0925`
- tick `130103`, seconds `72.00`, LSTM `0.3702`, delta `-0.0894`
- tick `131511`, seconds `94.00`, LSTM `0.4217`, delta `+0.0879`
- tick `131415`, seconds `92.50`, LSTM `0.3058`, delta `+0.0782`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005566`, |coef| `0.005566`
- `lag_00__CT_shots_fired_sum`: coefficient `0.004031`, |coef| `0.004031`
- `lag_00__CT_defusing_count`: coefficient `0.003837`, |coef| `0.003837`
- `lag_00__damage_diff_last_5s`: coefficient `0.003616`, |coef| `0.003616`
- `lag_05__CT_place_TRAMP`: coefficient `-0.003588`, |coef| `0.003588`
- `lag_00__T_kills_last_3s`: coefficient `-0.003534`, |coef| `0.003534`
- `lag_00__CT_place_BANANA`: coefficient `0.003466`, |coef| `0.003466`
- `lag_00__CT_kills_last_3s`: coefficient `0.003455`, |coef| `0.003455`
- `lag_02__CT_place_BALCONY`: coefficient `0.003019`, |coef| `0.003019`
- `lag_00__T_velocity_mean`: coefficient `-0.002938`, |coef| `0.002938`
- `lag_09__CT_place_TRAMP`: coefficient `0.002835`, |coef| `0.002835`
- `lag_15__CT_place_RUINS`: coefficient `0.002742`, |coef| `0.002742`
- `lag_01__T_duck_amount_mean`: coefficient `-0.002700`, |coef| `0.002700`
- `lag_01__CT_place_BANANA`: coefficient `0.002407`, |coef| `0.002407`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002402`, |coef| `0.002402`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002402` (lowers CT win probability)
- `lag_06__CT5__smoke`: coefficient `0.001797` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.001622` (lowers CT win probability)
- `lag_05__CT5__smoke`: coefficient `0.001579` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.001393` (lowers CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001332` (raises CT win probability)
- `lag_09__T4__smoke`: coefficient `-0.001243` (lowers CT win probability)
- `lag_11__T1__smoke`: coefficient `-0.001222` (lowers CT win probability)
- `lag_08__CT5__smoke`: coefficient `0.001169` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.001075` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005566` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.004031` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003837` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003616` (raises CT win probability)
- `lag_05__CT_place_TRAMP`: coefficient `-0.003588` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003534` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.003466` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003455` (raises CT win probability)
- `lag_02__CT_place_BALCONY`: coefficient `0.003019` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.002938` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `132279`, seconds `106.00`, LSTM delta `+0.2987`

Top all feature movements:
- `lag_05__CT_place_TRAMP`: contribution `+0.048345`
- `lag_09__CT_place_TRAMP`: contribution `+0.038191`
- `lag_00__kill_diff_last_3s`: contribution `+0.013396`
- `lag_00__CT_kills_last_3s`: contribution `+0.009976`
- `lag_15__T_bomb_zone_count`: contribution `+0.009789`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `+0.003932`

### tick `133495`, seconds `125.00`, LSTM delta `+0.2480`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.037196`
- `lag_00__CT_shots_fired_sum`: contribution `+0.016801`
- `lag_01__T_duck_amount_mean`: contribution `+0.015702`
- `lag_00__T_flash_alpha_mean`: contribution `+0.014571`
- `lag_00__kill_diff_last_3s`: contribution `+0.013396`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.014571`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.004577`

### tick `129719`, seconds `66.00`, LSTM delta `+0.2137`

Top all feature movements:
- `lag_02__CT_place_BALCONY`: contribution `+0.019373`
- `lag_00__CT_shots_fired_sum`: contribution `+0.016801`
- `lag_00__kill_diff_last_3s`: contribution `+0.013396`
- `lag_00__CT_kills_last_3s`: contribution `+0.009976`
- `lag_00__damage_diff_last_5s`: contribution `+0.008158`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `+0.004827`

### tick `130615`, seconds `80.00`, LSTM delta `-0.1434`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.022402`
- `lag_00__kill_diff_last_3s`: contribution `-0.013396`
- `lag_00__T_kills_last_3s`: contribution `-0.011196`
- `lag_08__CT_place_ARCH`: contribution `-0.008363`
- `lag_09__CT_place_ARCH`: contribution `-0.006516`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `133079`, seconds `118.50`, LSTM delta `-0.1204`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.013396`
- `lag_00__T_kills_last_3s`: contribution `-0.011196`
- `lag_00__CT_place_BANANA`: contribution `-0.010260`
- `lag_01__T_duck_amount_mean`: contribution `-0.005718`
- `lag_01__CT5__duck_amount`: contribution `-0.004682`

Top utility-only movements:
- `lag_06__CT5__smoke`: contribution `-0.003942`
