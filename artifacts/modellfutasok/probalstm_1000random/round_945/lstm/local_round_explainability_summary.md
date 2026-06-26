# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `91874`, seconds `74.00`, LSTM `0.8377`, delta `+0.2905`
- tick `91106`, seconds `62.00`, LSTM `0.2057`, delta `-0.2817`
- tick `91394`, seconds `66.50`, LSTM `0.3196`, delta `+0.2463`
- tick `89154`, seconds `31.50`, LSTM `0.5841`, delta `-0.1641`
- tick `88866`, seconds `27.00`, LSTM `0.7591`, delta `+0.1209`
- tick `90818`, seconds `57.50`, LSTM `0.4491`, delta `+0.1016`
- tick `91778`, seconds `72.50`, LSTM `0.4730`, delta `+0.0748`
- tick `91138`, seconds `62.50`, LSTM `0.1358`, delta `-0.0698`
- tick `91970`, seconds `75.50`, LSTM `0.9166`, delta `+0.0662`
- tick `89890`, seconds `43.00`, LSTM `0.4035`, delta `-0.0656`

## Top 15 local ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004795`, |coef| `0.004795`
- `lag_00__kill_diff_last_3s`: coefficient `0.003293`, |coef| `0.003293`
- `lag_09__CT4__flash_duration`: coefficient `-0.002954`, |coef| `0.002954`
- `lag_00__CT_kills_last_3s`: coefficient `0.002929`, |coef| `0.002929`
- `lag_09__CT5__flash_duration`: coefficient `-0.002747`, |coef| `0.002747`
- `lag_04__T_duck_amount_mean`: coefficient `0.002691`, |coef| `0.002691`
- `lag_09__CT_flash_duration_sum`: coefficient `-0.002567`, |coef| `0.002567`
- `lag_00__CT4__flash_duration`: coefficient `0.002536`, |coef| `0.002536`
- `lag_09__kill_diff_last_3s`: coefficient `-0.002292`, |coef| `0.002292`
- `lag_00__T_place_PALACEINTERIOR`: coefficient `-0.002275`, |coef| `0.002275`
- `lag_01__T_flash_alpha_mean`: coefficient `-0.002240`, |coef| `0.002240`
- `lag_15__kill_diff_last_3s`: coefficient `0.002198`, |coef| `0.002198`
- `lag_03__T_duck_amount_mean`: coefficient `-0.002150`, |coef| `0.002150`
- `lag_04__T4__duck_amount`: coefficient `0.002085`, |coef| `0.002085`
- `lag_15__CT_kills_last_3s`: coefficient `0.002042`, |coef| `0.002042`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004795` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.002954` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.002747` (lowers CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `-0.002567` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.002536` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.002240` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.002025` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001697` (raises CT win probability)
- `lag_09__CT5__smoke`: coefficient `-0.001657` (lowers CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `0.001551` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003293` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002929` (raises CT win probability)
- `lag_04__T_duck_amount_mean`: coefficient `0.002691` (raises CT win probability)
- `lag_09__kill_diff_last_3s`: coefficient `-0.002292` (lowers CT win probability)
- `lag_00__T_place_PALACEINTERIOR`: coefficient `-0.002275` (lowers CT win probability)
- `lag_15__kill_diff_last_3s`: coefficient `0.002198` (raises CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.002150` (lowers CT win probability)
- `lag_04__T4__duck_amount`: coefficient `0.002085` (raises CT win probability)
- `lag_15__CT_kills_last_3s`: coefficient `0.002042` (raises CT win probability)
- `lag_02__T4__duck_amount`: coefficient `-0.002035` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `91874`, seconds `74.00`, LSTM delta `+0.2905`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.029095`
- `lag_04__T_duck_amount_mean`: contribution `+0.015650`
- `lag_03__T_duck_amount_mean`: contribution `+0.012506`
- `lag_00__CT_kills_last_3s`: contribution `+0.008458`
- `lag_00__kill_diff_last_3s`: contribution `+0.007926`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.029095`

### tick `91106`, seconds `62.00`, LSTM delta `-0.2817`

Top all feature movements:
- `lag_09__CT4__flash_duration`: contribution `-0.021949`
- `lag_09__CT_flash_duration_sum`: contribution `-0.021178`
- `lag_09__CT5__flash_duration`: contribution `-0.020782`
- `lag_00__CT4__flash_duration`: contribution `-0.018846`
- `lag_09__CT_flashed_players`: contribution `-0.011715`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.021949`
- `lag_09__CT_flash_duration_sum`: contribution `-0.021178`
- `lag_09__CT5__flash_duration`: contribution `-0.020782`
- `lag_00__CT4__flash_duration`: contribution `-0.018846`
- `lag_09__T4__flash_duration`: contribution `-0.009961`

### tick `91394`, seconds `66.50`, LSTM delta `+0.2463`

Top all feature movements:
- `lag_09__CT4__flash_duration`: contribution `+0.021949`
- `lag_09__T4__flash_duration`: contribution `+0.009961`
- `lag_09__CT_flash_duration_sum`: contribution `+0.008531`
- `lag_00__CT_kills_last_3s`: contribution `+0.008458`
- `lag_13__CT_shots_fired_sum`: contribution `+0.008362`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `+0.021949`
- `lag_09__T4__flash_duration`: contribution `+0.009961`
- `lag_09__CT_flash_duration_sum`: contribution `+0.008531`
- `lag_04__CT5__flash_duration`: contribution `+0.007717`
- `lag_07__T1__flash_duration`: contribution `+0.007531`

### tick `89154`, seconds `31.50`, LSTM delta `-0.1641`

Top all feature movements:
- `lag_07__T_place_SCAFFOLDING`: contribution `-0.043843`
- `lag_06__T_place_SCAFFOLDING`: contribution `-0.037257`
- `lag_00__CT4__flash_duration`: contribution `+0.010293`
- `lag_00__kill_diff_last_3s`: contribution `-0.007926`
- `lag_13__CT_shots_fired_sum`: contribution `-0.006968`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.010293`
- `lag_02__CT4__flash_duration`: contribution `-0.005446`
- `lag_00__T4__flash_duration`: contribution `+0.003944`
- `lag_00__T_flash_duration_sum`: contribution `+0.003782`
- `lag_10__CT4__flash_duration`: contribution `-0.003745`

### tick `88866`, seconds `27.00`, LSTM delta `+0.1209`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `+0.009743`
- `lag_00__CT_kills_last_3s`: contribution `+0.008458`
- `lag_00__kill_diff_last_3s`: contribution `+0.007926`
- `lag_05__CT2__flash_duration`: contribution `+0.005940`
- `lag_08__CT5__shots_fired`: contribution `+0.004653`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `+0.005940`
- `lag_01__CT4__flash_duration`: contribution `+0.004447`
- `lag_05__T2__flash_duration`: contribution `+0.003421`
- `lag_08__CT1__flash_duration`: contribution `+0.003024`
- `lag_05__CT_flash_duration_sum`: contribution `+0.002649`
