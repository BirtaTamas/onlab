# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `7`

## Largest probability jumps

- tick `59739`, seconds `82.50`, LSTM `0.7773`, delta `+0.3267`
- tick `59483`, seconds `78.50`, LSTM `0.3523`, delta `+0.2558`
- tick `59515`, seconds `79.00`, LSTM `0.6077`, delta `+0.2554`
- tick `59675`, seconds `81.50`, LSTM `0.4540`, delta `-0.2408`
- tick `58235`, seconds `59.00`, LSTM `0.0978`, delta `-0.1571`
- tick `60219`, seconds `90.00`, LSTM `0.7605`, delta `-0.1477`
- tick `60091`, seconds `88.00`, LSTM `0.8873`, delta `+0.1163`
- tick `58075`, seconds `56.50`, LSTM `0.4340`, delta `-0.1158`
- tick `58203`, seconds `58.50`, LSTM `0.2549`, delta `-0.0760`
- tick `58107`, seconds `57.00`, LSTM `0.3641`, delta `-0.0699`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004524`, |coef| `0.004524`
- `lag_04__CT_shots_fired_sum`: coefficient `0.003991`, |coef| `0.003991`
- `lag_00__CT_kills_last_3s`: coefficient `0.003678`, |coef| `0.003678`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003455`, |coef| `0.003455`
- `lag_06__CT_shots_fired_sum`: coefficient `-0.003450`, |coef| `0.003450`
- `lag_04__CT2__shots_fired`: coefficient `0.003157`, |coef| `0.003157`
- `lag_00__damage_diff_last_5s`: coefficient `0.003002`, |coef| `0.003002`
- `lag_01__CT_place_SNIPERSNEST`: coefficient `-0.002771`, |coef| `0.002771`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `-0.002770`, |coef| `0.002770`
- `lag_06__CT2__shots_fired`: coefficient `-0.002764`, |coef| `0.002764`
- `lag_04__T_place_CONNECTOR`: coefficient `0.002708`, |coef| `0.002708`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002576`, |coef| `0.002576`
- `lag_08__T_place_CONNECTOR`: coefficient `0.002560`, |coef| `0.002560`
- `lag_00__CT_damage_last_5s`: coefficient `0.002388`, |coef| `0.002388`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002322`, |coef| `0.002322`

## Top 10 utility ridge features

- `lag_02__T_B_site_active_smokes`: coefficient `-0.001350` (lowers CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `-0.001262` (lowers CT win probability)
- `lag_14__CT_B_site_active_smokes`: coefficient `-0.001197` (lowers CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `-0.001077` (lowers CT win probability)
- `lag_01__T_B_site_active_smokes`: coefficient `-0.001060` (lowers CT win probability)
- `lag_02__T_active_smokes`: coefficient `-0.000978` (lowers CT win probability)
- `lag_04__T_B_site_active_smokes`: coefficient `-0.000832` (lowers CT win probability)
- `lag_03__T_active_smokes`: coefficient `-0.000815` (lowers CT win probability)
- `lag_01__T_active_smokes`: coefficient `-0.000803` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.000783` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004524` (raises CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `0.003991` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003678` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003455` (raises CT win probability)
- `lag_06__CT_shots_fired_sum`: coefficient `-0.003450` (lowers CT win probability)
- `lag_04__CT2__shots_fired`: coefficient `0.003157` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003002` (raises CT win probability)
- `lag_01__CT_place_SNIPERSNEST`: coefficient `-0.002771` (lowers CT win probability)
- `lag_00__CT_place_SNIPERSNEST`: coefficient `-0.002770` (lowers CT win probability)
- `lag_06__CT2__shots_fired`: coefficient `-0.002764` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `59739`, seconds `82.50`, LSTM delta `+0.3267`

Top all feature movements:
- `lag_06__CT_shots_fired_sum`: contribution `+0.050335`
- `lag_06__CT2__shots_fired`: contribution `+0.028857`
- `lag_03__CT_place_BRICKS`: contribution `+0.023298`
- `lag_08__T_place_CONNECTOR`: contribution `+0.012398`
- `lag_00__kill_diff_last_3s`: contribution `+0.010890`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `59483`, seconds `78.50`, LSTM delta `+0.2558`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.014837`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012000`
- `lag_02__T_place_CONNECTOR`: contribution `+0.010915`
- `lag_00__kill_diff_last_3s`: contribution `+0.010890`
- `lag_00__CT_kills_last_3s`: contribution `+0.010620`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `59515`, seconds `79.00`, LSTM delta `+0.2554`

Top all feature movements:
- `lag_01__CT_place_SNIPERSNEST`: contribution `+0.014842`
- `lag_04__T_place_CONNECTOR`: contribution `+0.013115`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012000`
- `lag_00__kill_diff_last_3s`: contribution `+0.010890`
- `lag_00__CT_kills_last_3s`: contribution `+0.010620`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `59675`, seconds `81.50`, LSTM delta `-0.2408`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.058220`
- `lag_04__CT2__shots_fired`: contribution `-0.032954`
- `lag_00__kill_diff_last_3s`: contribution `-0.021780`
- `lag_08__T_place_CONNECTOR`: contribution `-0.012398`
- `lag_06__CT_shots_fired_sum`: contribution `-0.011985`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58235`, seconds `59.00`, LSTM delta `-0.1571`

Top all feature movements:
- `lag_04__CT_place_TUNNEL`: contribution `-0.023789`
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `-0.013391`
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `-0.011764`
- `lag_00__kill_diff_last_3s`: contribution `-0.010890`
- `lag_00__T_kills_last_3s`: contribution `-0.006079`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.004599`
