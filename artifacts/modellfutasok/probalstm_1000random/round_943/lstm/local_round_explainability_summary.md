# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `29`

## Largest probability jumps

- tick `243601`, seconds `65.00`, LSTM `0.7886`, delta `+0.1373`
- tick `245809`, seconds `99.50`, LSTM `0.9264`, delta `+0.1173`
- tick `243217`, seconds `59.00`, LSTM `0.5877`, delta `+0.0832`
- tick `246609`, seconds `112.00`, LSTM `0.9647`, delta `+0.0369`
- tick `243057`, seconds `56.50`, LSTM `0.4967`, delta `-0.0325`
- tick `243121`, seconds `57.50`, LSTM `0.5045`, delta `+0.0315`
- tick `244433`, seconds `78.00`, LSTM `0.8377`, delta `+0.0286`
- tick `244145`, seconds `73.50`, LSTM `0.8135`, delta `+0.0285`
- tick `245201`, seconds `90.00`, LSTM `0.8419`, delta `+0.0274`
- tick `239793`, seconds `5.50`, LSTM `0.6172`, delta `-0.0261`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001464`, |coef| `0.001464`
- `lag_00__T_burning_players`: coefficient `-0.001304`, |coef| `0.001304`
- `lag_12__T_place_SIDEHALL`: coefficient `-0.001072`, |coef| `0.001072`
- `lag_00__kill_diff_last_3s`: coefficient `0.001024`, |coef| `0.001024`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.001022`, |coef| `0.001022`
- `lag_00__T4__alive`: coefficient `-0.001001`, |coef| `0.001001`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `-0.000986`, |coef| `0.000986`
- `lag_03__T_place_SIDEENTRANCE`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_00__CT_damage_last_5s`: coefficient `0.000952`, |coef| `0.000952`
- `lag_00__T4__armor`: coefficient `-0.000934`, |coef| `0.000934`
- `lag_04__CT5__is_walking`: coefficient `-0.000912`, |coef| `0.000912`
- `lag_06__T_burning_players`: coefficient `0.000911`, |coef| `0.000911`
- `lag_00__CT_walking_count`: coefficient `-0.000903`, |coef| `0.000903`
- `lag_00__T4__hp`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_00__damage_diff_last_5s`: coefficient `0.000869`, |coef| `0.000869`

## Top 10 utility ridge features

- `lag_00__T4__flash`: coefficient `-0.000661` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000540` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000502` (lowers CT win probability)
- `lag_01__T_he_last_5s`: coefficient `0.000493` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000489` (lowers CT win probability)
- `lag_12__T2__flash`: coefficient `-0.000489` (lowers CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.000475` (raises CT win probability)
- `lag_00__T_he_last_5s`: coefficient `0.000467` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000456` (lowers CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `0.000454` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001464` (raises CT win probability)
- `lag_00__T_burning_players`: coefficient `-0.001304` (lowers CT win probability)
- `lag_12__T_place_SIDEHALL`: coefficient `-0.001072` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001024` (raises CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.001022` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.001001` (lowers CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `-0.000986` (lowers CT win probability)
- `lag_03__T_place_SIDEENTRANCE`: coefficient `-0.000972` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000952` (raises CT win probability)
- `lag_00__T4__armor`: coefficient `-0.000934` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `243601`, seconds `65.00`, LSTM delta `+0.1373`

Top all feature movements:
- `lag_12__T_place_SIDEHALL`: contribution `+0.013902`
- `lag_00__CT_kills_last_3s`: contribution `+0.004227`
- `lag_00__T_place_HOUSE`: contribution `+0.002733`
- `lag_00__kill_diff_last_3s`: contribution `+0.002464`
- `lag_10__CT3__duck_amount`: contribution `+0.002388`

Top utility-only movements:
- `lag_15__T_utility_damage_last_5s`: contribution `+0.001629`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.001555`
- `lag_12__T2__flash`: contribution `+0.001439`

### tick `245809`, seconds `99.50`, LSTM delta `+0.1173`

Top all feature movements:
- `lag_03__T_place_SIDEENTRANCE`: contribution `+0.004746`
- `lag_00__CT_kills_last_3s`: contribution `+0.004227`
- `lag_00__T_burning_players`: contribution `+0.003303`
- `lag_00__kill_diff_last_3s`: contribution `+0.002464`
- `lag_00__T4__alive`: contribution `+0.002461`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `243217`, seconds `59.00`, LSTM delta `+0.0832`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.013246`
- `lag_06__T_burning_players`: contribution `+0.006924`
- `lag_00__T_burning_players`: contribution `+0.003303`
- `lag_13__T_place_SIDEHALL`: contribution `+0.003247`
- `lag_03__T_place_SIDEHALL`: contribution `+0.002781`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.001439`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.001195`

### tick `246609`, seconds `112.00`, LSTM delta `+0.0369`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004227`
- `lag_10__CT_flashed_players`: contribution `-0.003807`
- `lag_08__CT_flashed_players`: contribution `+0.003157`
- `lag_00__T_flash_alpha_mean`: contribution `+0.002769`
- `lag_00__kill_diff_last_3s`: contribution `+0.002464`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.002769`

### tick `243057`, seconds `56.50`, LSTM delta `-0.0325`

Top all feature movements:
- `lag_01__T_burning_players`: contribution `-0.006544`
- `lag_02__T_place_SIDEHALL`: contribution `-0.003495`
- `lag_06__T_place_SIDEHALL`: contribution `-0.002359`
- `lag_15__CT_place_ALLEY`: contribution `+0.001856`
- `lag_09__T_place_SIDEHALL`: contribution `-0.001795`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.001038`
