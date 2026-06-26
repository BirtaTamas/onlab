# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `17`

## Largest probability jumps

- tick `144115`, seconds `27.50`, LSTM `0.1032`, delta `-0.3192`
- tick `143059`, seconds `11.00`, LSTM `0.3821`, delta `-0.2081`
- tick `146739`, seconds `68.50`, LSTM `0.0131`, delta `-0.1124`
- tick `146707`, seconds `68.00`, LSTM `0.1256`, delta `+0.0969`
- tick `143379`, seconds `16.00`, LSTM `0.4187`, delta `+0.0577`
- tick `142451`, seconds `1.50`, LSTM `0.5977`, delta `-0.0551`
- tick `142739`, seconds `6.00`, LSTM `0.6028`, delta `+0.0546`
- tick `143091`, seconds `11.50`, LSTM `0.3307`, delta `-0.0514`
- tick `145011`, seconds `41.50`, LSTM `0.0252`, delta `-0.0498`
- tick `142771`, seconds `6.50`, LSTM `0.5531`, delta `-0.0497`

## Top 15 local ridge features

- `lag_02__CT5__flash_duration`: coefficient `-0.002028`, |coef| `0.002028`
- `lag_00__T_kills_last_3s`: coefficient `-0.002002`, |coef| `0.002002`
- `lag_15__CT_place_BDOORS`: coefficient `-0.001942`, |coef| `0.001942`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001937`, |coef| `0.001937`
- `lag_02__T4__is_scoped`: coefficient `-0.001859`, |coef| `0.001859`
- `lag_00__damage_diff_last_5s`: coefficient `0.001839`, |coef| `0.001839`
- `lag_00__kill_diff_last_3s`: coefficient `0.001829`, |coef| `0.001829`
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.001804`, |coef| `0.001804`
- `lag_00__CT_place_BDOORS`: coefficient `0.001762`, |coef| `0.001762`
- `lag_00__CT5__flash_duration`: coefficient `0.001747`, |coef| `0.001747`
- `lag_09__T_mollies_last_5s`: coefficient `0.001689`, |coef| `0.001689`
- `lag_00__T_damage_last_5s`: coefficient `-0.001577`, |coef| `0.001577`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001501`, |coef| `0.001501`
- `lag_00__CT5__alive`: coefficient `0.001477`, |coef| `0.001477`
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.001460`, |coef| `0.001460`

## Top 10 utility ridge features

- `lag_02__CT5__flash_duration`: coefficient `-0.002028` (lowers CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.001804` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001747` (raises CT win probability)
- `lag_09__T_mollies_last_5s`: coefficient `0.001689` (raises CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.001460` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001345` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.001332` (raises CT win probability)
- `lag_10__T_smokes_last_5s`: coefficient `0.001209` (raises CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.001082` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001078` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002002` (lowers CT win probability)
- `lag_15__CT_place_BDOORS`: coefficient `-0.001942` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001937` (lowers CT win probability)
- `lag_02__T4__is_scoped`: coefficient `-0.001859` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001839` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001829` (raises CT win probability)
- `lag_00__CT_place_BDOORS`: coefficient `0.001762` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001577` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001501` (lowers CT win probability)
- `lag_00__CT5__alive`: coefficient `0.001477` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `144115`, seconds `27.50`, LSTM delta `-0.3192`

Top all feature movements:
- `lag_02__CT5__flash_duration`: contribution `-0.015472`
- `lag_00__T_shots_fired_sum`: contribution `-0.014522`
- `lag_00__CT5__flash_duration`: contribution `-0.013328`
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.009531`
- `lag_15__CT_place_BDOORS`: contribution `-0.009341`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `-0.015472`
- `lag_00__CT5__flash_duration`: contribution `-0.013328`
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.009531`
- `lag_06__utility_damage_diff_last_5s`: contribution `-0.006326`

### tick `143059`, seconds `11.00`, LSTM delta `-0.2081`

Top all feature movements:
- `lag_09__T_mollies_last_5s`: contribution `-0.034733`
- `lag_10__T_smokes_last_5s`: contribution `-0.017728`
- `lag_10__T_he_last_5s`: contribution `-0.014118`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `-0.007434`
- `lag_09__T_flashes_last_5s`: contribution `-0.006914`

Top utility-only movements:
- `lag_09__T_mollies_last_5s`: contribution `-0.034733`
- `lag_10__T_smokes_last_5s`: contribution `-0.017728`
- `lag_10__T_he_last_5s`: contribution `-0.014118`
- `lag_09__T_flashes_last_5s`: contribution `-0.006914`

### tick `146739`, seconds `68.50`, LSTM delta `-0.1124`

Top all feature movements:
- `lag_01__T4__is_scoped`: contribution `-0.006605`
- `lag_00__T_kills_last_3s`: contribution `-0.006344`
- `lag_00__T_shots_fired_sum`: contribution `-0.005809`
- `lag_08__CT_place_EXTENDEDA`: contribution `-0.004944`
- `lag_00__kill_diff_last_3s`: contribution `-0.004403`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `146707`, seconds `68.00`, LSTM delta `+0.0969`

Top all feature movements:
- `lag_06__T4__is_scoped`: contribution `+0.004878`
- `lag_00__kill_diff_last_3s`: contribution `+0.004403`
- `lag_15__CT_place_ARAMP`: contribution `+0.003312`
- `lag_01__T4__duck_amount`: contribution `+0.003118`
- `lag_00__damage_diff_last_5s`: contribution `+0.002739`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `143379`, seconds `16.00`, LSTM delta `+0.0577`

Top all feature movements:
- `lag_15__CT_place_BDOORS`: contribution `+0.009341`
- `lag_07__T_place_TUNNELSTAIRS`: contribution `+0.004677`
- `lag_01__CT_place_SHORTSTAIRS`: contribution `+0.004245`
- `lag_00__damage_diff_last_5s`: contribution `+0.004149`
- `lag_00__T_damage_last_5s`: contribution `+0.003782`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `+0.001209`
