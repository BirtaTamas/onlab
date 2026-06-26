# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `13158`, seconds `32.50`, LSTM `0.2365`, delta `-0.2777`
- tick `13446`, seconds `37.00`, LSTM `0.2030`, delta `+0.0747`
- tick `13926`, seconds `44.50`, LSTM `0.1815`, delta `-0.0746`
- tick `14502`, seconds `53.50`, LSTM `0.0150`, delta `-0.0724`
- tick `13958`, seconds `45.00`, LSTM `0.1145`, delta `-0.0669`
- tick `14022`, seconds `46.00`, LSTM `0.0304`, delta `-0.0581`
- tick `13478`, seconds `37.50`, LSTM `0.2481`, delta `+0.0451`
- tick `11686`, seconds `9.50`, LSTM `0.5202`, delta `+0.0448`
- tick `13798`, seconds `42.50`, LSTM `0.2793`, delta `+0.0427`
- tick `11814`, seconds `11.50`, LSTM `0.4558`, delta `-0.0383`

## Top 15 local ridge features

- `lag_00__CT_place_LOWERTUNNEL`: coefficient `0.002385`, |coef| `0.002385`
- `lag_11__CT_place_UPPERTUNNEL`: coefficient `0.002053`, |coef| `0.002053`
- `lag_01__T1__flash_duration`: coefficient `-0.001824`, |coef| `0.001824`
- `lag_14__T_place_LOWERTUNNEL`: coefficient `-0.001569`, |coef| `0.001569`
- `lag_00__T_kills_last_3s`: coefficient `-0.001494`, |coef| `0.001494`
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.001377`, |coef| `0.001377`
- `lag_05__CT_utility_damage_last_5s`: coefficient `-0.001375`, |coef| `0.001375`
- `lag_14__CT_place_LONGDOORS`: coefficient `-0.001328`, |coef| `0.001328`
- `lag_13__CT_place_EXTENDEDA`: coefficient `0.001320`, |coef| `0.001320`
- `lag_00__T_damage_last_5s`: coefficient `-0.001299`, |coef| `0.001299`
- `lag_11__CT_place_UNDERA`: coefficient `-0.001282`, |coef| `0.001282`
- `lag_00__damage_diff_last_5s`: coefficient `0.001244`, |coef| `0.001244`
- `lag_05__T_place_TUNNELSTAIRS`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_03__T_shots_fired_sum`: coefficient `-0.001229`, |coef| `0.001229`
- `lag_01__T_flashed_players`: coefficient `-0.001208`, |coef| `0.001208`

## Top 10 utility ridge features

- `lag_01__T1__flash_duration`: coefficient `-0.001824` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `-0.001375` (lowers CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.001157` (lowers CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `-0.001150` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.001048` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000999` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000983` (raises CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `-0.000970` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000833` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000833` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_LOWERTUNNEL`: coefficient `0.002385` (raises CT win probability)
- `lag_11__CT_place_UPPERTUNNEL`: coefficient `0.002053` (raises CT win probability)
- `lag_14__T_place_LOWERTUNNEL`: coefficient `-0.001569` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001494` (lowers CT win probability)
- `lag_14__T_place_TUNNELSTAIRS`: coefficient `0.001377` (raises CT win probability)
- `lag_14__CT_place_LONGDOORS`: coefficient `-0.001328` (lowers CT win probability)
- `lag_13__CT_place_EXTENDEDA`: coefficient `0.001320` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001299` (lowers CT win probability)
- `lag_11__CT_place_UNDERA`: coefficient `-0.001282` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001244` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `13158`, seconds `32.50`, LSTM delta `-0.2777`

Top all feature movements:
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.017529`
- `lag_11__CT_place_UPPERTUNNEL`: contribution `-0.015744`
- `lag_01__T1__flash_duration`: contribution `-0.009627`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `-0.009613`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.008654`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `-0.009627`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.004539`
- `lag_01__T_flash_duration_sum`: contribution `-0.003449`

### tick `13446`, seconds `37.00`, LSTM delta `+0.0747`

Top all feature movements:
- `lag_01__T1__flash_duration`: contribution `+0.009627`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `+0.009613`
- `lag_14__T_place_LOWERTUNNEL`: contribution `+0.006783`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.005692`
- `lag_04__T_place_TUNNELSTAIRS`: contribution `+0.004073`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `+0.009627`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.003819`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.002626`
- `lag_01__T_flash_duration_sum`: contribution `+0.002301`
- `lag_10__T1__flash_duration`: contribution `+0.001238`

### tick `13926`, seconds `44.50`, LSTM delta `-0.0746`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.005528`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.005474`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.003765`
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.003728`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `-0.003370`

Top utility-only movements:
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.005474`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.003765`

### tick `14502`, seconds `53.50`, LSTM delta `-0.0724`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.004733`
- `lag_14__T_shots_fired_sum`: contribution `-0.003995`
- `lag_15__T_shots_fired_sum`: contribution `-0.003798`
- `lag_00__T_damage_last_5s`: contribution `-0.003085`
- `lag_00__damage_diff_last_5s`: contribution `-0.002778`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.002141`

### tick `13958`, seconds `45.00`, LSTM delta `-0.0669`

Top all feature movements:
- `lag_00__T2__shots_fired`: contribution `-0.009207`
- `lag_13__CT_place_EXTENDEDA`: contribution `-0.007409`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.006506`
- `lag_00__T_shots_fired_sum`: contribution `+0.005216`
- `lag_00__T_place_LOWERTUNNEL`: contribution `-0.004800`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.006506`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.004464`
- `lag_00__CT4__flash_duration`: contribution `-0.001859`
