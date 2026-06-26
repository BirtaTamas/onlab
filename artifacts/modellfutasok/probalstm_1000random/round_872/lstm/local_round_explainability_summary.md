# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `102814`, seconds `87.50`, LSTM `0.2801`, delta `-0.2626`
- tick `102974`, seconds `90.00`, LSTM `0.0544`, delta `-0.2004`
- tick `103166`, seconds `93.00`, LSTM `0.0275`, delta `-0.1990`
- tick `103038`, seconds `91.00`, LSTM `0.1662`, delta `+0.1285`
- tick `103134`, seconds `92.50`, LSTM `0.2265`, delta `+0.0616`
- tick `102942`, seconds `89.50`, LSTM `0.2548`, delta `+0.0497`
- tick `99390`, seconds `34.00`, LSTM `0.5404`, delta `+0.0431`
- tick `102878`, seconds `88.50`, LSTM `0.2219`, delta `-0.0347`
- tick `99486`, seconds `35.50`, LSTM `0.5659`, delta `+0.0259`
- tick `101534`, seconds `67.50`, LSTM `0.5609`, delta `+0.0255`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002259`, |coef| `0.002259`
- `lag_00__T_damage_last_5s`: coefficient `-0.002127`, |coef| `0.002127`
- `lag_02__T_place_MIDDOORS`: coefficient `-0.001943`, |coef| `0.001943`
- `lag_05__T_place_MIDDOORS`: coefficient `-0.001894`, |coef| `0.001894`
- `lag_00__T_kills_last_3s`: coefficient `-0.001875`, |coef| `0.001875`
- `lag_01__CT_place_BDOORS`: coefficient `0.001848`, |coef| `0.001848`
- `lag_07__T_place_TUNNELSTAIRS`: coefficient `0.001795`, |coef| `0.001795`
- `lag_00__kill_diff_last_3s`: coefficient `0.001751`, |coef| `0.001751`
- `lag_13__CT_place_BDOORS`: coefficient `-0.001699`, |coef| `0.001699`
- `lag_00__CT_place_MIDDOORS`: coefficient `0.001698`, |coef| `0.001698`
- `lag_05__T_place_TUNNELSTAIRS`: coefficient `0.001669`, |coef| `0.001669`
- `lag_07__CT2__is_scoped`: coefficient `-0.001618`, |coef| `0.001618`
- `lag_06__CT1__duck_amount`: coefficient `0.001566`, |coef| `0.001566`
- `lag_02__CT2__is_scoped`: coefficient `-0.001554`, |coef| `0.001554`
- `lag_00__CT1__duck_amount`: coefficient `0.001505`, |coef| `0.001505`

## Top 10 utility ridge features

- `lag_00__CT5__smoke`: coefficient `0.001016` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000932` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000880` (raises CT win probability)
- `lag_05__CT2__flash`: coefficient `0.000801` (raises CT win probability)
- `lag_07__CT_active_infernos`: coefficient `0.000790` (raises CT win probability)
- `lag_12__T2__flash`: coefficient `-0.000731` (lowers CT win probability)
- `lag_04__T3__molly`: coefficient `-0.000715` (lowers CT win probability)
- `lag_13__T3__flash`: coefficient `0.000710` (raises CT win probability)
- `lag_05__T1__flash`: coefficient `0.000684` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.000653` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002259` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002127` (lowers CT win probability)
- `lag_02__T_place_MIDDOORS`: coefficient `-0.001943` (lowers CT win probability)
- `lag_05__T_place_MIDDOORS`: coefficient `-0.001894` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001875` (lowers CT win probability)
- `lag_01__CT_place_BDOORS`: coefficient `0.001848` (raises CT win probability)
- `lag_07__T_place_TUNNELSTAIRS`: coefficient `0.001795` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001751` (raises CT win probability)
- `lag_13__CT_place_BDOORS`: coefficient `-0.001699` (lowers CT win probability)
- `lag_00__CT_place_MIDDOORS`: coefficient `0.001698` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `102814`, seconds `87.50`, LSTM delta `-0.2626`

Top all feature movements:
- `lag_07__T_place_TUNNELSTAIRS`: contribution `-0.012531`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `-0.011655`
- `lag_11__T_place_TUNNELSTAIRS`: contribution `-0.010075`
- `lag_02__CT2__is_scoped`: contribution `-0.009509`
- `lag_15__T2__is_scoped`: contribution `-0.009459`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `102974`, seconds `90.00`, LSTM delta `-0.2004`

Top all feature movements:
- `lag_07__CT2__is_scoped`: contribution `-0.009901`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.008593`
- `lag_07__T_place_MIDDOORS`: contribution `-0.006086`
- `lag_00__T_kills_last_3s`: contribution `-0.005941`
- `lag_00__CT1__duck_amount`: contribution `-0.005740`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103166`, seconds `93.00`, LSTM delta `-0.1990`

Top all feature movements:
- `lag_06__CT1__duck_amount`: contribution `-0.005974`
- `lag_02__T2__is_scoped`: contribution `-0.005727`
- `lag_13__CT2__is_scoped`: contribution `-0.005548`
- `lag_03__T4__duck_amount`: contribution `-0.005113`
- `lag_00__T_damage_last_5s`: contribution `-0.005100`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103038`, seconds `91.00`, LSTM delta `+0.1285`

Top all feature movements:
- `lag_06__CT1__duck_amount`: contribution `+0.005974`
- `lag_03__CT1__duck_amount`: contribution `+0.005282`
- `lag_00__damage_diff_last_5s`: contribution `+0.005096`
- `lag_08__CT2__duck_amount`: contribution `+0.005036`
- `lag_00__kill_diff_last_3s`: contribution `+0.004215`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103134`, seconds `92.50`, LSTM delta `+0.0616`

Top all feature movements:
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.008593`
- `lag_06__CT1__duck_amount`: contribution `+0.005974`
- `lag_00__T_damage_last_5s`: contribution `+0.005100`
- `lag_00__damage_diff_last_5s`: contribution `+0.005096`
- `lag_02__T4__duck_amount`: contribution `+0.003332`

Top utility-only movements:
- No utility movement among the top local contributors.
