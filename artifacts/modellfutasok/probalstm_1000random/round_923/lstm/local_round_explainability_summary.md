# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `19`

## Largest probability jumps

- tick `149260`, seconds `34.50`, LSTM `0.9296`, delta `+0.0670`
- tick `149196`, seconds `33.50`, LSTM `0.8511`, delta `+0.0590`
- tick `147820`, seconds `12.00`, LSTM `0.7188`, delta `+0.0333`
- tick `147468`, seconds `6.50`, LSTM `0.7325`, delta `+0.0300`
- tick `147724`, seconds `10.50`, LSTM `0.7013`, delta `-0.0270`
- tick `147884`, seconds `13.00`, LSTM `0.7617`, delta `+0.0251`
- tick `148364`, seconds `20.50`, LSTM `0.7747`, delta `+0.0240`
- tick `149996`, seconds `46.00`, LSTM `0.9613`, delta `+0.0236`
- tick `148428`, seconds `21.50`, LSTM `0.7591`, delta `-0.0197`
- tick `147852`, seconds `12.50`, LSTM `0.7365`, delta `+0.0178`

## Top 15 local ridge features

- `lag_07__CT_place_JUNGLE`: coefficient `0.000992`, |coef| `0.000992`
- `lag_00__CT_kills_last_3s`: coefficient `0.000685`, |coef| `0.000685`
- `lag_08__CT_place_JUNGLE`: coefficient `0.000678`, |coef| `0.000678`
- `lag_06__CT_place_JUNGLE`: coefficient `0.000676`, |coef| `0.000676`
- `lag_11__CT_place_STAIRS`: coefficient `0.000658`, |coef| `0.000658`
- `lag_09__CT_place_JUNGLE`: coefficient `0.000656`, |coef| `0.000656`
- `lag_00__CT_place_TRUCK`: coefficient `0.000633`, |coef| `0.000633`
- `lag_00__kill_diff_last_3s`: coefficient `0.000597`, |coef| `0.000597`
- `lag_07__T2__duck_amount`: coefficient `0.000485`, |coef| `0.000485`
- `lag_09__CT_place_STAIRS`: coefficient `0.000463`, |coef| `0.000463`
- `lag_00__damage_diff_last_5s`: coefficient `0.000450`, |coef| `0.000450`
- `lag_00__T5__alive`: coefficient `-0.000443`, |coef| `0.000443`
- `lag_00__T3__alive`: coefficient `-0.000432`, |coef| `0.000432`
- `lag_00__CT_place_CTSPAWN`: coefficient `-0.000431`, |coef| `0.000431`
- `lag_00__CT_damage_last_5s`: coefficient `0.000425`, |coef| `0.000425`

## Top 10 utility ridge features

- `lag_08__CT4__smoke`: coefficient `-0.000293` (lowers CT win probability)
- `lag_03__T1__molly`: coefficient `-0.000291` (lowers CT win probability)
- `lag_05__T1__molly`: coefficient `-0.000288` (lowers CT win probability)
- `lag_10__CT4__smoke`: coefficient `-0.000283` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `-0.000270` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000262` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000248` (lowers CT win probability)
- `lag_04__T2__molly`: coefficient `0.000247` (raises CT win probability)
- `lag_09__CT4__smoke`: coefficient `-0.000242` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `-0.000237` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_JUNGLE`: coefficient `0.000992` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000685` (raises CT win probability)
- `lag_08__CT_place_JUNGLE`: coefficient `0.000678` (raises CT win probability)
- `lag_06__CT_place_JUNGLE`: coefficient `0.000676` (raises CT win probability)
- `lag_11__CT_place_STAIRS`: coefficient `0.000658` (raises CT win probability)
- `lag_09__CT_place_JUNGLE`: coefficient `0.000656` (raises CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.000633` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000597` (raises CT win probability)
- `lag_07__T2__duck_amount`: coefficient `0.000485` (raises CT win probability)
- `lag_09__CT_place_STAIRS`: coefficient `0.000463` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `149260`, seconds `34.50`, LSTM delta `+0.0670`

Top all feature movements:
- `lag_11__CT_place_STAIRS`: contribution `+0.005124`
- `lag_09__CT_place_JUNGLE`: contribution `+0.004207`
- `lag_02__CT_place_TRUCK`: contribution `+0.002111`
- `lag_00__CT_kills_last_3s`: contribution `+0.001978`
- `lag_07__T2__duck_amount`: contribution `+0.001854`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `149196`, seconds `33.50`, LSTM delta `+0.0590`

Top all feature movements:
- `lag_07__CT_place_JUNGLE`: contribution `+0.006365`
- `lag_00__CT_place_TRUCK`: contribution `+0.004080`
- `lag_09__CT_place_STAIRS`: contribution `+0.003606`
- `lag_00__CT_kills_last_3s`: contribution `+0.001978`
- `lag_07__T2__duck_amount`: contribution `+0.001854`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `147820`, seconds `12.00`, LSTM delta `+0.0333`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `+0.004080`
- `lag_05__CT_place_JUNGLE`: contribution `+0.002213`
- `lag_14__CT_place_SHOP`: contribution `+0.001441`
- `lag_05__T2__duck_amount`: contribution `+0.001334`
- `lag_02__CT2__duck_amount`: contribution `+0.001082`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.000982`

### tick `147468`, seconds `6.50`, LSTM delta `+0.0300`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.002186`
- `lag_03__CT_place_SHOP`: contribution `+0.001719`
- `lag_02__CT_place_SHOP`: contribution `+0.001643`
- `lag_04__T_place_PALACEALLEY`: contribution `+0.000867`
- `lag_03__T_place_PALACEINTERIOR`: contribution `+0.000847`

Top utility-only movements:
- `lag_00__T1__smoke`: contribution `+0.000536`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.000353`
- `lag_00__CT_active_infernos`: contribution `-0.000298`

### tick `147724`, seconds `10.50`, LSTM delta `-0.0270`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.002186`
- `lag_08__CT_place_SNIPERSNEST`: contribution `-0.001554`
- `lag_06__CT_place_SHOP`: contribution `-0.001436`
- `lag_05__CT_place_SHOP`: contribution `-0.001425`
- `lag_01__T2__duck_amount`: contribution `+0.001385`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.000741`
- `lag_00__CT3__molly`: contribution `-0.000560`
- `lag_03__T_A_site_active_infernos`: contribution `-0.000503`
- `lag_03__T_B_site_active_infernos`: contribution `-0.000460`
