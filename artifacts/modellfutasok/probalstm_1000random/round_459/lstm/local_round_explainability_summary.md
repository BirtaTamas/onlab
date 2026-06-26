# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `22`

## Largest probability jumps

- tick `184447`, seconds `73.50`, LSTM `0.5229`, delta `+0.3240`
- tick `184479`, seconds `74.00`, LSTM `0.2042`, delta `-0.3187`
- tick `184127`, seconds `68.50`, LSTM `0.3127`, delta `-0.2388`
- tick `184255`, seconds `70.50`, LSTM `0.1830`, delta `-0.2105`
- tick `184511`, seconds `74.50`, LSTM `0.0285`, delta `-0.1758`
- tick `184223`, seconds `70.00`, LSTM `0.3935`, delta `+0.0899`
- tick `184159`, seconds `69.00`, LSTM `0.2706`, delta `-0.0421`
- tick `184191`, seconds `69.50`, LSTM `0.3036`, delta `+0.0330`
- tick `182335`, seconds `40.50`, LSTM `0.5958`, delta `-0.0293`
- tick `182207`, seconds `38.50`, LSTM `0.5980`, delta `-0.0289`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002776`, |coef| `0.002776`
- `lag_06__T4__flash_duration`: coefficient `-0.002612`, |coef| `0.002612`
- `lag_00__damage_diff_last_5s`: coefficient `0.002484`, |coef| `0.002484`
- `lag_00__T_damage_last_5s`: coefficient `-0.002218`, |coef| `0.002218`
- `lag_00__CT2__flash_duration`: coefficient `0.001861`, |coef| `0.001861`
- `lag_00__kill_diff_last_3s`: coefficient `0.001731`, |coef| `0.001731`
- `lag_06__CT_place_HOLE`: coefficient `-0.001649`, |coef| `0.001649`
- `lag_10__T_shots_fired_sum`: coefficient `0.001641`, |coef| `0.001641`
- `lag_08__CT_place_ARAMP`: coefficient `-0.001598`, |coef| `0.001598`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.001555`, |coef| `0.001555`
- `lag_11__CT_place_HOLE`: coefficient `-0.001516`, |coef| `0.001516`
- `lag_10__CT_place_BDOORS`: coefficient `-0.001509`, |coef| `0.001509`
- `lag_10__T4__shots_fired`: coefficient `0.001472`, |coef| `0.001472`
- `lag_11__CT2__flash_duration`: coefficient `0.001449`, |coef| `0.001449`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001354`, |coef| `0.001354`

## Top 10 utility ridge features

- `lag_06__T4__flash_duration`: coefficient `-0.002612` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001861` (raises CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.001449` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.001273` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.001072` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.001063` (lowers CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `0.001018` (raises CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `0.001002` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000867` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000854` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002776` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002484` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002218` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001731` (raises CT win probability)
- `lag_06__CT_place_HOLE`: coefficient `-0.001649` (lowers CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `0.001641` (raises CT win probability)
- `lag_08__CT_place_ARAMP`: coefficient `-0.001598` (lowers CT win probability)
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.001555` (lowers CT win probability)
- `lag_11__CT_place_HOLE`: coefficient `-0.001516` (lowers CT win probability)
- `lag_10__CT_place_BDOORS`: coefficient `-0.001509` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `184447`, seconds `73.50`, LSTM delta `+0.3240`

Top all feature movements:
- `lag_06__CT_place_HOLE`: contribution `+0.018408`
- `lag_01__T_place_SHORTSTAIRS`: contribution `+0.013071`
- `lag_00__damage_diff_last_5s`: contribution `+0.011098`
- `lag_08__CT_place_ARAMP`: contribution `+0.009953`
- `lag_10__CT2__flash_duration`: contribution `+0.009766`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `+0.009766`
- `lag_06__T4__flash_duration`: contribution `+0.008411`
- `lag_14__CT2__flash_duration`: contribution `+0.005804`
- `lag_00__T5__flash_duration`: contribution `+0.004369`
- `lag_05__T4__flash_duration`: contribution `-0.003336`

### tick `184479`, seconds `74.00`, LSTM delta `-0.3187`

Top all feature movements:
- `lag_06__T4__flash_duration`: contribution `-0.015899`
- `lag_07__CT_place_HOLE`: contribution `+0.012566`
- `lag_11__CT2__flash_duration`: contribution `-0.011113`
- `lag_02__T_place_SHORTSTAIRS`: contribution `-0.010861`
- `lag_00__T_kills_last_3s`: contribution `-0.008793`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.015899`
- `lag_11__CT2__flash_duration`: contribution `-0.011113`
- `lag_06__T5__flash_duration`: contribution `-0.005357`
- `lag_15__CT2__flash_duration`: contribution `-0.004216`
- `lag_01__T5__flash_duration`: contribution `-0.004126`

### tick `184127`, seconds `68.50`, LSTM delta `-0.2388`

Top all feature movements:
- `lag_00__CT2__flash_duration`: contribution `-0.014275`
- `lag_07__CT_place_HOLE`: contribution `-0.012566`
- `lag_00__T_kills_last_3s`: contribution `-0.008793`
- `lag_10__CT_place_BDOORS`: contribution `-0.007261`
- `lag_00__T_shots_fired_sum`: contribution `-0.007107`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.014275`
- `lag_04__CT2__flash_duration`: contribution `-0.003518`

### tick `184255`, seconds `70.50`, LSTM delta `-0.2105`

Top all feature movements:
- `lag_11__CT_place_HOLE`: contribution `-0.016927`
- `lag_00__T_kills_last_3s`: contribution `-0.008793`
- `lag_00__CT_place_HOLE`: contribution `-0.007902`
- `lag_00__T_damage_last_5s`: contribution `-0.005318`
- `lag_02__CT_place_ARAMP`: contribution `-0.005146`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.004660`
- `lag_08__CT2__flash_duration`: contribution `-0.004154`
- `lag_09__T4__flash_duration`: contribution `-0.002993`

### tick `184511`, seconds `74.50`, LSTM delta `-0.1758`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008793`
- `lag_00__T_shots_fired_sum`: contribution `+0.008122`
- `lag_12__CT2__flash_duration`: contribution `-0.007804`
- `lag_03__T_place_SHORTSTAIRS`: contribution `-0.006763`
- `lag_01__T_shots_fired_sum`: contribution `-0.006179`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `-0.007804`
- `lag_07__T4__flash_duration`: contribution `-0.004906`
- `lag_00__CT1__flash`: contribution `-0.002926`
- `lag_02__T5__flash_duration`: contribution `-0.002652`
- `lag_07__T5__flash_duration`: contribution `-0.002648`
