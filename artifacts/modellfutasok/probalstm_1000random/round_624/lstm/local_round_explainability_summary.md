# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `19`

## Largest probability jumps

- tick `183740`, seconds `75.00`, LSTM `0.7854`, delta `+0.2809`
- tick `183708`, seconds `74.50`, LSTM `0.5045`, delta `+0.2375`
- tick `183772`, seconds `75.50`, LSTM `0.9311`, delta `+0.1456`
- tick `184284`, seconds `83.50`, LSTM `0.9614`, delta `+0.0800`
- tick `182460`, seconds `55.00`, LSTM `0.3410`, delta `-0.0705`
- tick `183580`, seconds `72.50`, LSTM `0.3038`, delta `-0.0580`
- tick `182780`, seconds `60.00`, LSTM `0.3781`, delta `+0.0475`
- tick `179740`, seconds `12.50`, LSTM `0.3889`, delta `+0.0280`
- tick `183804`, seconds `76.00`, LSTM `0.9577`, delta `+0.0266`
- tick `182556`, seconds `56.50`, LSTM `0.3060`, delta `-0.0254`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.004358`, |coef| `0.004358`
- `lag_00__kill_diff_last_3s`: coefficient `0.003743`, |coef| `0.003743`
- `lag_00__T_flashed_players`: coefficient `-0.003266`, |coef| `0.003266`
- `lag_02__T_place_UPPERTUNNEL`: coefficient `-0.003031`, |coef| `0.003031`
- `lag_00__damage_diff_last_5s`: coefficient `0.002960`, |coef| `0.002960`
- `lag_00__CT_damage_last_5s`: coefficient `0.002780`, |coef| `0.002780`
- `lag_05__CT_flashed_players`: coefficient `0.002616`, |coef| `0.002616`
- `lag_01__damage_diff_last_5s`: coefficient `0.002562`, |coef| `0.002562`
- `lag_01__CT_kills_last_3s`: coefficient `0.002414`, |coef| `0.002414`
- `lag_04__CT4__flash_duration`: coefficient `0.002403`, |coef| `0.002403`
- `lag_01__CT_damage_last_5s`: coefficient `0.002397`, |coef| `0.002397`
- `lag_01__T_place_UPPERTUNNEL`: coefficient `-0.002204`, |coef| `0.002204`
- `lag_05__CT4__flash_duration`: coefficient `0.002195`, |coef| `0.002195`
- `lag_04__CT_flashed_players`: coefficient `0.002156`, |coef| `0.002156`
- `lag_00__T_macro_B`: coefficient `-0.002137`, |coef| `0.002137`

## Top 10 utility ridge features

- `lag_04__CT4__flash_duration`: coefficient `0.002403` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.002195` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.001744` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.001709` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001659` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.001605` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001583` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.001447` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `-0.001432` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.001414` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.004358` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003743` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.003266` (lowers CT win probability)
- `lag_02__T_place_UPPERTUNNEL`: coefficient `-0.003031` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002960` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002780` (raises CT win probability)
- `lag_05__CT_flashed_players`: coefficient `0.002616` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002562` (raises CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.002414` (raises CT win probability)
- `lag_01__CT_damage_last_5s`: coefficient `0.002397` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `183740`, seconds `75.00`, LSTM delta `+0.2809`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `+0.012606`
- `lag_00__CT_kills_last_3s`: contribution `+0.012581`
- `lag_05__CT_flashed_players`: contribution `+0.011459`
- `lag_05__CT4__flash_duration`: contribution `+0.010121`
- `lag_00__kill_diff_last_3s`: contribution `+0.009010`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `+0.010121`
- `lag_02__T3__flash_duration`: contribution `+0.006546`
- `lag_00__T3__flash_duration`: contribution `+0.006246`

### tick `183708`, seconds `74.50`, LSTM delta `+0.2375`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `+0.012606`
- `lag_00__CT_kills_last_3s`: contribution `+0.012581`
- `lag_04__CT4__flash_duration`: contribution `+0.011078`
- `lag_04__CT_flashed_players`: contribution `+0.009443`
- `lag_00__kill_diff_last_3s`: contribution `+0.009010`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.011078`
- `lag_01__T3__flash_duration`: contribution `+0.004893`
- `lag_00__T5__smoke`: contribution `+0.003478`

### tick `183772`, seconds `75.50`, LSTM delta `+0.1456`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.012581`
- `lag_00__kill_diff_last_3s`: contribution `+0.009010`
- `lag_01__CT_kills_last_3s`: contribution `+0.006970`
- `lag_00__damage_diff_last_5s`: contribution `+0.006678`
- `lag_04__T_flashed_players`: contribution `-0.006549`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.004893`
- `lag_06__CT4__flash_duration`: contribution `+0.004246`
- `lag_03__T3__flash_duration`: contribution `+0.003227`

### tick `184284`, seconds `83.50`, LSTM delta `+0.0800`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.012581`
- `lag_00__kill_diff_last_3s`: contribution `+0.009010`
- `lag_00__damage_diff_last_5s`: contribution `+0.006010`
- `lag_00__CT_damage_last_5s`: contribution `+0.005454`
- `lag_14__T1__flash_duration`: contribution `+0.003940`

Top utility-only movements:
- `lag_14__T1__flash_duration`: contribution `+0.003940`
- `lag_01__CT4__flash_duration`: contribution `+0.001640`

### tick `182460`, seconds `55.00`, LSTM delta `-0.0705`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.015488`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.008554`
- `lag_02__T_place_UPPERTUNNEL`: contribution `-0.006969`
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.005493`
- `lag_00__CT_place_ARAMP`: contribution `-0.003532`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.015488`
