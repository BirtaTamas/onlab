# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `17`

## Largest probability jumps

- tick `123295`, seconds `61.50`, LSTM `0.7128`, delta `+0.3170`
- tick `123327`, seconds `62.00`, LSTM `0.5137`, delta `-0.1991`
- tick `121279`, seconds `30.00`, LSTM `0.7544`, delta `+0.1683`
- tick `122079`, seconds `42.50`, LSTM `0.5190`, delta `-0.1512`
- tick `124159`, seconds `75.00`, LSTM `0.4025`, delta `+0.1311`
- tick `121119`, seconds `27.50`, LSTM `0.5423`, delta `-0.1246`
- tick `124063`, seconds `73.50`, LSTM `0.2645`, delta `-0.0823`
- tick `121887`, seconds `39.50`, LSTM `0.7013`, delta `-0.0737`
- tick `123903`, seconds `71.00`, LSTM `0.3218`, delta `-0.0708`
- tick `123679`, seconds `67.50`, LSTM `0.4421`, delta `-0.0708`

## Top 15 local ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.002597`, |coef| `0.002597`
- `lag_03__T_place_UNDERA`: coefficient `0.002550`, |coef| `0.002550`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002416`, |coef| `0.002416`
- `lag_12__T_place_ARAMP`: coefficient `0.002320`, |coef| `0.002320`
- `lag_00__T_place_UNDERA`: coefficient `-0.002295`, |coef| `0.002295`
- `lag_15__CT_place_SHORTSTAIRS`: coefficient `-0.002176`, |coef| `0.002176`
- `lag_14__T_bomb_zone_count`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_00__kill_diff_last_3s`: coefficient `0.001830`, |coef| `0.001830`
- `lag_01__CT5__is_scoped`: coefficient `0.001753`, |coef| `0.001753`
- `lag_04__T_place_LONGA`: coefficient `0.001650`, |coef| `0.001650`
- `lag_12__T_place_UNDERA`: coefficient `-0.001604`, |coef| `0.001604`
- `lag_00__CT_defusing_count`: coefficient `0.001578`, |coef| `0.001578`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001438`, |coef| `0.001438`
- `lag_03__CT4__flash_duration`: coefficient `-0.001422`, |coef| `0.001422`
- `lag_12__CT4__flash_duration`: coefficient `0.001422`, |coef| `0.001422`

## Top 10 utility ridge features

- `lag_03__CT4__flash_duration`: coefficient `-0.001422` (lowers CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `0.001422` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.001391` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001360` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `0.001290` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `0.001193` (raises CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.001151` (lowers CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001145` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.001122` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.000984` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_ARAMP`: coefficient `-0.002597` (lowers CT win probability)
- `lag_03__T_place_UNDERA`: coefficient `0.002550` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002416` (lowers CT win probability)
- `lag_12__T_place_ARAMP`: coefficient `0.002320` (raises CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `-0.002295` (lowers CT win probability)
- `lag_15__CT_place_SHORTSTAIRS`: coefficient `-0.002176` (lowers CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `-0.001850` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001830` (raises CT win probability)
- `lag_01__CT5__is_scoped`: coefficient `0.001753` (raises CT win probability)
- `lag_04__T_place_LONGA`: coefficient `0.001650` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `123295`, seconds `61.50`, LSTM delta `+0.3170`

Top all feature movements:
- `lag_00__T_place_ARAMP`: contribution `+0.023497`
- `lag_12__T_place_ARAMP`: contribution `+0.020993`
- `lag_14__T_bomb_zone_count`: contribution `+0.010767`
- `lag_12__CT_flash_duration_sum`: contribution `+0.008008`
- `lag_02__T_flashed_players`: contribution `+0.007969`

Top utility-only movements:
- `lag_12__CT_flash_duration_sum`: contribution `+0.008008`
- `lag_01__CT5__flash_duration`: contribution `+0.007889`
- `lag_02__CT2__flash_duration`: contribution `+0.007474`
- `lag_03__CT4__flash_duration`: contribution `+0.007415`
- `lag_12__CT4__flash_duration`: contribution `+0.007413`

### tick `123327`, seconds `62.00`, LSTM delta `-0.1991`

Top all feature movements:
- `lag_15__CT_place_SHORTSTAIRS`: contribution `-0.012130`
- `lag_03__CT4__flash_duration`: contribution `-0.007119`
- `lag_01__CT5__is_scoped`: contribution `-0.006268`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.006131`
- `lag_15__T_bomb_zone_count`: contribution `-0.005690`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.007119`
- `lag_00__CT4__flash_duration`: contribution `-0.004235`
- `lag_03__CT_flash_duration_sum`: contribution `-0.003606`
- `lag_02__CT_flash_duration_sum`: contribution `-0.003558`

### tick `121279`, seconds `30.00`, LSTM delta `+0.1683`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.015764`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `+0.012130`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `+0.007764`
- `lag_14__CT1__flash_duration`: contribution `+0.005915`
- `lag_07__T5__flash_duration`: contribution `+0.005659`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `+0.005915`
- `lag_07__T5__flash_duration`: contribution `+0.005659`
- `lag_12__T5__flash_duration`: contribution `+0.005014`
- `lag_09__T1__flash_duration`: contribution `+0.004459`
- `lag_02__CT_flash_duration_sum`: contribution `+0.004439`

### tick `122079`, seconds `42.50`, LSTM delta `-0.1512`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.010868`
- `lag_06__T_place_ARAMP`: contribution `-0.010802`
- `lag_11__T_place_OUTSIDETUNNEL`: contribution `-0.005236`
- `lag_01__T_shots_fired_sum`: contribution `-0.004931`
- `lag_00__kill_diff_last_3s`: contribution `-0.004405`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `-0.003371`

### tick `124159`, seconds `75.00`, LSTM delta `+0.1311`

Top all feature movements:
- `lag_03__T_place_UNDERA`: contribution `+0.039854`
- `lag_00__CT_defusing_count`: contribution `+0.015299`
- `lag_15__CT_shots_fired_sum`: contribution `+0.011634`
- `lag_07__CT_place_EXTENDEDA`: contribution `+0.006568`
- `lag_08__T_place_PIT`: contribution `+0.006516`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.001405`
