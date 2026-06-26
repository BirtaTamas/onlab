# Random LSTM Round Suite

- rounds: `100`
- LSTM round wins by MAE: `57`
- XGBoost round wins by MAE: `43`
- LSTM closer ticks total: `11092`
- XGBoost closer ticks total: `7600`

## Selected Rounds

| idx | rows | round_num | csv |
|---:|---:|---:|---|
| 1 | 153 | 20 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv` |
| 2 | 203 | 15 | `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv` |
| 3 | 197 | 5 | `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv` |
| 4 | 182 | 5 | `processed_full/iem_katowice/iem-katowice-2025-spirit-vs-astralis-bo3-GZVTrKsE-zdG9dH6juITei/spirit-vs-astralis-m1-nuke.csv` |
| 5 | 230 | 11 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv` |
| 6 | 272 | 3 | `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv` |
| 7 | 217 | 17 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv` |
| 8 | 162 | 5 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv` |
| 9 | 122 | 11 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv` |
| 10 | 146 | 1 | `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv` |
| 11 | 240 | 20 | `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv` |
| 12 | 225 | 8 | `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv` |
| 13 | 158 | 9 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv` |
| 14 | 102 | 8 | `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv` |
| 15 | 248 | 22 | `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv` |
| 16 | 117 | 16 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv` |
| 17 | 253 | 6 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv` |
| 18 | 102 | 1 | `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv` |
| 19 | 152 | 5 | `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv` |
| 20 | 137 | 4 | `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv` |
| 21 | 140 | 4 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv` |
| 22 | 252 | 9 | `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv` |
| 23 | 217 | 6 | `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv` |
| 24 | 188 | 9 | `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv` |
| 25 | 230 | 10 | `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv` |
| 26 | 196 | 2 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv` |
| 27 | 167 | 7 | `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv` |
| 28 | 237 | 14 | `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv` |
| 29 | 218 | 4 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv` |
| 30 | 222 | 15 | `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv` |
| 31 | 122 | 12 | `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv` |
| 32 | 158 | 6 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv` |
| 33 | 175 | 8 | `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv` |
| 34 | 234 | 11 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv` |
| 35 | 148 | 5 | `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv` |
| 36 | 183 | 26 | `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv` |
| 37 | 108 | 13 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv` |
| 38 | 157 | 8 | `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m2-dust2.csv` |
| 39 | 215 | 14 | `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv` |
| 40 | 216 | 4 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv` |
| 41 | 138 | 1 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-nrg-dust2-QDtqFlW1Z9UhZpBNOAavnd/heroic-vs-nrg-dust2.csv` |
| 42 | 281 | 17 | `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv` |
| 43 | 156 | 3 | `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv` |
| 44 | 187 | 10 | `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv` |
| 45 | 251 | 16 | `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv` |
| 46 | 223 | 27 | `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv` |
| 47 | 177 | 20 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv` |
| 48 | 223 | 5 | `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv` |
| 49 | 138 | 8 | `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv` |
| 50 | 111 | 16 | `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv` |
| 51 | 152 | 4 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv` |
| 52 | 265 | 4 | `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv` |
| 53 | 127 | 29 | `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv` |
| 54 | 198 | 9 | `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv` |
| 55 | 280 | 8 | `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv` |
| 56 | 159 | 14 | `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv` |
| 57 | 208 | 7 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv` |
| 58 | 230 | 18 | `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv` |
| 59 | 162 | 13 | `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv` |
| 60 | 113 | 13 | `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv` |
| 61 | 215 | 8 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv` |
| 62 | 166 | 16 | `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv` |
| 63 | 186 | 15 | `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv` |
| 64 | 225 | 12 | `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv` |
| 65 | 231 | 5 | `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv` |
| 66 | 155 | 11 | `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv` |
| 67 | 157 | 10 | `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv` |
| 68 | 198 | 11 | `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv` |
| 69 | 216 | 10 | `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv` |
| 70 | 195 | 6 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv` |
| 71 | 243 | 2 | `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv` |
| 72 | 230 | 5 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv` |
| 73 | 176 | 21 | `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv` |
| 74 | 253 | 5 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv` |
| 75 | 204 | 15 | `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv` |
| 76 | 208 | 19 | `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv` |
| 77 | 203 | 15 | `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv` |
| 78 | 167 | 7 | `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv` |
| 79 | 306 | 5 | `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv` |
| 80 | 262 | 15 | `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv` |
| 81 | 134 | 8 | `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv` |
| 82 | 200 | 10 | `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv` |
| 83 | 223 | 2 | `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv` |
| 84 | 161 | 1 | `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv` |
| 85 | 125 | 6 | `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv` |
| 86 | 199 | 15 | `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv` |
| 87 | 225 | 4 | `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv` |
| 88 | 158 | 10 | `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-Z9VnvF_JkEDX6y_HyMsFXx/aurora-vs-heroic-m3-mirage.csv` |
| 89 | 133 | 9 | `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv` |
| 90 | 209 | 11 | `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv` |
| 91 | 145 | 12 | `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv` |
| 92 | 170 | 15 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv` |
| 93 | 177 | 3 | `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv` |
| 94 | 150 | 4 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv` |
| 95 | 124 | 1 | `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv` |
| 96 | 139 | 13 | `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv` |
| 97 | 123 | 14 | `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv` |
| 98 | 205 | 10 | `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv` |
| 99 | 154 | 9 | `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv` |
| 100 | 162 | 11 | `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv` |

## Model Comparison

| idx | true_ct_win | rows | winner | lstm_mae | xgb_mae | lstm_logloss | xgb_logloss | lstm_closer | xgb_closer |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 153 | lstm | 0.604246 | 0.604924 | 1.070539 | 1.057217 | 92 | 61 |
| 2 | 1 | 203 | lstm | 0.481704 | 0.519737 | 0.668992 | 0.763499 | 133 | 70 |
| 3 | 0 | 197 | xgboost | 0.194228 | 0.192647 | 0.280641 | 0.261956 | 136 | 61 |
| 4 | 0 | 182 | xgboost | 0.417420 | 0.397520 | 0.600238 | 0.557039 | 61 | 121 |
| 5 | 1 | 230 | xgboost | 0.265991 | 0.254868 | 0.341100 | 0.326585 | 82 | 148 |
| 6 | 0 | 272 | lstm | 0.247659 | 0.299609 | 0.335129 | 0.422930 | 267 | 5 |
| 7 | 0 | 217 | xgboost | 0.410235 | 0.384582 | 0.606270 | 0.503876 | 82 | 135 |
| 8 | 0 | 162 | lstm | 0.008677 | 0.040943 | 0.008748 | 0.042277 | 137 | 25 |
| 9 | 0 | 122 | xgboost | 0.453326 | 0.434787 | 0.679099 | 0.647933 | 57 | 65 |
| 10 | 1 | 146 | xgboost | 0.350485 | 0.333500 | 0.463948 | 0.443118 | 68 | 78 |
| 11 | 0 | 240 | lstm | 0.096696 | 0.175509 | 0.106980 | 0.206996 | 240 | 0 |
| 12 | 1 | 225 | xgboost | 0.259387 | 0.215265 | 0.327032 | 0.268772 | 10 | 215 |
| 13 | 0 | 158 | lstm | 0.558411 | 0.616715 | 0.880820 | 1.027355 | 149 | 9 |
| 14 | 0 | 102 | lstm | 0.208544 | 0.289213 | 0.274634 | 0.421457 | 90 | 12 |
| 15 | 0 | 248 | xgboost | 0.551472 | 0.438718 | 0.860678 | 0.635873 | 68 | 180 |
| 16 | 0 | 117 | lstm | 0.125130 | 0.181355 | 0.144577 | 0.218813 | 116 | 1 |
| 17 | 1 | 253 | xgboost | 0.388663 | 0.283966 | 0.576853 | 0.354175 | 65 | 188 |
| 18 | 1 | 102 | xgboost | 0.319345 | 0.266550 | 0.402849 | 0.333692 | 26 | 76 |
| 19 | 0 | 152 | lstm | 0.190625 | 0.198920 | 0.280576 | 0.290097 | 128 | 24 |
| 20 | 1 | 137 | lstm | 0.165737 | 0.251549 | 0.202276 | 0.314074 | 124 | 13 |
| 21 | 1 | 140 | xgboost | 0.085902 | 0.018469 | 0.090749 | 0.018698 | 0 | 140 |
| 22 | 1 | 252 | xgboost | 0.811620 | 0.711395 | 2.837315 | 1.631557 | 2 | 250 |
| 23 | 1 | 217 | xgboost | 0.288659 | 0.233334 | 0.354579 | 0.272310 | 69 | 148 |
| 24 | 0 | 188 | lstm | 0.158198 | 0.220980 | 0.218449 | 0.297794 | 182 | 6 |
| 25 | 1 | 230 | xgboost | 0.298792 | 0.259109 | 0.375284 | 0.317983 | 35 | 195 |
| 26 | 1 | 196 | xgboost | 0.266435 | 0.249371 | 0.321051 | 0.296327 | 49 | 147 |
| 27 | 0 | 167 | lstm | 0.176891 | 0.249182 | 0.231122 | 0.337241 | 161 | 6 |
| 28 | 0 | 237 | lstm | 0.221316 | 0.264245 | 0.271385 | 0.331584 | 221 | 16 |
| 29 | 1 | 218 | xgboost | 0.127914 | 0.119968 | 0.157522 | 0.158300 | 49 | 169 |
| 30 | 1 | 222 | lstm | 0.604916 | 0.677002 | 1.473280 | 1.577659 | 135 | 87 |
| 31 | 1 | 122 | xgboost | 0.173388 | 0.131077 | 0.216172 | 0.168087 | 10 | 112 |
| 32 | 0 | 158 | lstm | 0.538667 | 0.610519 | 1.105882 | 1.603529 | 153 | 5 |
| 33 | 0 | 175 | lstm | 0.101000 | 0.188995 | 0.123574 | 0.237177 | 175 | 0 |
| 34 | 0 | 234 | xgboost | 0.224459 | 0.188702 | 0.316567 | 0.253341 | 68 | 166 |
| 35 | 1 | 148 | xgboost | 0.399590 | 0.292419 | 0.531590 | 0.365117 | 1 | 147 |
| 36 | 1 | 183 | xgboost | 0.275865 | 0.192191 | 0.342290 | 0.229719 | 8 | 175 |
| 37 | 1 | 108 | xgboost | 0.316943 | 0.279107 | 0.413072 | 0.355510 | 18 | 90 |
| 38 | 1 | 157 | xgboost | 0.441494 | 0.391818 | 0.606830 | 0.520908 | 2 | 155 |
| 39 | 0 | 215 | lstm | 0.130312 | 0.132059 | 0.152859 | 0.157042 | 152 | 63 |
| 40 | 0 | 216 | xgboost | 0.312204 | 0.249596 | 0.473863 | 0.335433 | 100 | 116 |
| 41 | 0 | 138 | lstm | 0.259808 | 0.335034 | 0.335188 | 0.457347 | 125 | 13 |
| 42 | 0 | 281 | xgboost | 0.135059 | 0.131463 | 0.174267 | 0.167737 | 205 | 76 |
| 43 | 0 | 156 | lstm | 0.011600 | 0.040637 | 0.011743 | 0.041906 | 156 | 0 |
| 44 | 1 | 187 | xgboost | 0.163830 | 0.162363 | 0.182302 | 0.181617 | 82 | 105 |
| 45 | 1 | 251 | xgboost | 0.394208 | 0.381276 | 0.580526 | 0.548035 | 111 | 140 |
| 46 | 1 | 223 | xgboost | 0.445381 | 0.394220 | 1.103730 | 0.635658 | 108 | 115 |
| 47 | 0 | 177 | lstm | 0.299705 | 0.382863 | 0.410137 | 0.569299 | 177 | 0 |
| 48 | 0 | 223 | lstm | 0.470874 | 0.541033 | 0.744820 | 0.914774 | 217 | 6 |
| 49 | 1 | 138 | xgboost | 0.380297 | 0.377914 | 0.566902 | 0.546674 | 56 | 82 |
| 50 | 1 | 111 | xgboost | 0.386601 | 0.335438 | 0.520819 | 0.433315 | 4 | 107 |
| 51 | 0 | 152 | lstm | 0.401238 | 0.463110 | 0.544081 | 0.645326 | 121 | 31 |
| 52 | 0 | 265 | lstm | 0.244937 | 0.259719 | 0.320082 | 0.339801 | 192 | 73 |
| 53 | 1 | 127 | lstm | 0.420967 | 0.480726 | 0.578845 | 0.734674 | 101 | 26 |
| 54 | 0 | 198 | lstm | 0.367673 | 0.376023 | 0.633892 | 0.638044 | 127 | 71 |
| 55 | 0 | 280 | lstm | 0.039737 | 0.097988 | 0.042090 | 0.109308 | 280 | 0 |
| 56 | 0 | 159 | lstm | 0.459460 | 0.503989 | 0.993868 | 1.116342 | 144 | 15 |
| 57 | 0 | 208 | lstm | 0.525053 | 0.687549 | 0.811075 | 1.275070 | 208 | 0 |
| 58 | 1 | 230 | lstm | 0.641463 | 0.661925 | 1.420849 | 1.364083 | 168 | 62 |
| 59 | 0 | 162 | lstm | 0.138369 | 0.223791 | 0.156308 | 0.270937 | 162 | 0 |
| 60 | 0 | 113 | lstm | 0.249848 | 0.324418 | 0.333027 | 0.422652 | 81 | 32 |
| 61 | 1 | 215 | lstm | 0.424266 | 0.466818 | 0.557985 | 0.639148 | 196 | 19 |
| 62 | 0 | 166 | lstm | 0.270967 | 0.343244 | 0.340771 | 0.444688 | 150 | 16 |
| 63 | 0 | 186 | lstm | 0.012515 | 0.038771 | 0.012808 | 0.040673 | 186 | 0 |
| 64 | 0 | 225 | lstm | 0.188623 | 0.233242 | 0.231968 | 0.285927 | 199 | 26 |
| 65 | 0 | 231 | lstm | 0.179379 | 0.300962 | 0.225103 | 0.375846 | 197 | 34 |
| 66 | 0 | 155 | lstm | 0.434467 | 0.667165 | 0.621662 | 1.370995 | 154 | 1 |
| 67 | 1 | 157 | xgboost | 0.543967 | 0.464703 | 1.118507 | 0.692985 | 24 | 133 |
| 68 | 1 | 198 | xgboost | 0.638368 | 0.550496 | 1.298699 | 0.987123 | 17 | 181 |
| 69 | 1 | 216 | xgboost | 0.547638 | 0.506161 | 0.965852 | 0.766304 | 73 | 143 |
| 70 | 0 | 195 | lstm | 0.244477 | 0.253583 | 0.344976 | 0.347460 | 136 | 59 |
| 71 | 1 | 243 | xgboost | 0.431048 | 0.358258 | 0.631817 | 0.468285 | 9 | 234 |
| 72 | 1 | 230 | xgboost | 0.260618 | 0.179152 | 0.371102 | 0.228673 | 0 | 230 |
| 73 | 0 | 176 | lstm | 0.531147 | 0.584899 | 0.851343 | 0.961433 | 153 | 23 |
| 74 | 0 | 253 | lstm | 0.161359 | 0.208582 | 0.191276 | 0.249973 | 213 | 40 |
| 75 | 1 | 204 | lstm | 0.343867 | 0.376473 | 0.438728 | 0.500725 | 146 | 58 |
| 76 | 0 | 208 | lstm | 0.380477 | 0.453188 | 0.523714 | 0.660213 | 208 | 0 |
| 77 | 0 | 203 | lstm | 0.217220 | 0.231566 | 0.278905 | 0.299424 | 132 | 71 |
| 78 | 0 | 167 | lstm | 0.476756 | 0.557968 | 0.767573 | 0.986943 | 160 | 7 |
| 79 | 0 | 306 | lstm | 0.431176 | 0.568577 | 0.727821 | 1.119750 | 304 | 2 |
| 80 | 0 | 262 | xgboost | 0.355682 | 0.328386 | 0.522046 | 0.462213 | 88 | 174 |
| 81 | 1 | 134 | xgboost | 0.355928 | 0.270749 | 0.497868 | 0.353328 | 0 | 134 |
| 82 | 1 | 200 | lstm | 0.195950 | 0.290126 | 0.235462 | 0.381801 | 169 | 31 |
| 83 | 0 | 223 | lstm | 0.146985 | 0.161216 | 0.175653 | 0.191166 | 172 | 51 |
| 84 | 0 | 161 | lstm | 0.290914 | 0.336467 | 0.408655 | 0.498025 | 161 | 0 |
| 85 | 0 | 125 | lstm | 0.365179 | 0.367522 | 0.494473 | 0.478332 | 56 | 69 |
| 86 | 1 | 199 | xgboost | 0.129474 | 0.128340 | 0.140417 | 0.140084 | 97 | 102 |
| 87 | 0 | 225 | lstm | 0.360765 | 0.439024 | 0.501206 | 0.656284 | 195 | 30 |
| 88 | 1 | 158 | xgboost | 0.046221 | 0.018568 | 0.047396 | 0.018758 | 0 | 158 |
| 89 | 1 | 133 | lstm | 0.297452 | 0.298206 | 0.367587 | 0.376471 | 64 | 69 |
| 90 | 1 | 209 | xgboost | 0.379792 | 0.358934 | 0.494000 | 0.458885 | 39 | 170 |
| 91 | 0 | 145 | lstm | 0.487027 | 0.594333 | 0.694583 | 0.976992 | 140 | 5 |
| 92 | 1 | 170 | lstm | 0.267636 | 0.354705 | 0.317007 | 0.450960 | 160 | 10 |
| 93 | 0 | 177 | lstm | 0.174615 | 0.271490 | 0.201820 | 0.334084 | 170 | 7 |
| 94 | 1 | 150 | xgboost | 0.261811 | 0.240193 | 0.316191 | 0.288189 | 66 | 84 |
| 95 | 0 | 124 | xgboost | 0.392812 | 0.365062 | 0.548489 | 0.492261 | 27 | 97 |
| 96 | 0 | 139 | lstm | 0.515650 | 0.550133 | 0.732957 | 0.849203 | 54 | 85 |
| 97 | 1 | 123 | xgboost | 0.059767 | 0.014756 | 0.061910 | 0.014887 | 0 | 123 |
| 98 | 1 | 205 | lstm | 0.196278 | 0.205776 | 0.238350 | 0.275070 | 78 | 127 |
| 99 | 0 | 154 | lstm | 0.011175 | 0.026065 | 0.011320 | 0.026559 | 153 | 1 |
| 100 | 1 | 162 | xgboost | 0.482870 | 0.379464 | 0.671456 | 0.483089 | 0 | 162 |

## Utility Cohorts Across Random Rounds

| cohort | rows | rounds | lstm_mean_prob | xgb_mean_prob | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 18692 | 100 | 0.439903 | 0.474473 | 11092 | 7600 | 0.756420 | 0.749358 |
| active/recent utility | 18617 | 100 | 0.440553 | 0.474782 | 11017 | 7600 | 0.755439 | 0.748348 |
| strong utility action | 13624 | 100 | 0.467785 | 0.499271 | 7954 | 5670 | 0.739357 | 0.726585 |
| utility damage | 1251 | 74 | 0.519537 | 0.535805 | 676 | 575 | 0.687450 | 0.705835 |
| active smoke/inferno | 13273 | 100 | 0.467579 | 0.499120 | 7708 | 5565 | 0.737588 | 0.726512 |
| recent utility last 5s | 555 | 45 | 0.507467 | 0.539642 | 358 | 197 | 0.754955 | 0.632432 |
| flash effect present | 18603 | 100 | 0.440767 | 0.474847 | 11003 | 7600 | 0.755255 | 0.748159 |

## Frequent Top Ridge Features

| feature csalad | utility | round_presence | mean_abs_coef | max_abs_coef |
|---|---:|---:|---:|---:|
| `kill_diff_last_3s` | False | 73 | 0.002784 | 0.009495 |
| `kills_last_3s` | False | 71 | 0.002473 | 0.007331 |
| `damage_diff_last_5s` | False | 46 | 0.002405 | 0.006017 |
| `damage_last_5s` | False | 30 | 0.002042 | 0.004162 |
| `flash_duration` | True | 28 | 0.002007 | 0.004587 |
| `duck_amount` | False | 28 | 0.001856 | 0.003676 |
| `CT_shots_fired_sum` | False | 27 | 0.001881 | 0.004024 |
| `is_walking` | False | 15 | 0.001537 | 0.002820 |
| `T_shots_fired_sum` | False | 13 | 0.002144 | 0.006615 |
| `shots_fired` | False | 13 | 0.001361 | 0.002711 |
| `is_scoped` | False | 9 | 0.001602 | 0.003163 |
| `CT_velocity_mean` | False | 8 | 0.002077 | 0.005137 |
| `alive` | False | 8 | 0.002042 | 0.003234 |
| `CT_defusing_count` | False | 7 | 0.004903 | 0.006629 |
| `T_bomb_zone_count` | False | 7 | 0.002466 | 0.003927 |
| `flashed_players` | False | 7 | 0.001783 | 0.003436 |
| `flashes_last_5s` | True | 7 | 0.001462 | 0.002386 |
| `CT_place_BALCONY` | False | 6 | 0.001764 | 0.002381 |
| `CT_place_HUT` | False | 6 | 0.001673 | 0.002433 |
| `CT_place_VENTS` | False | 6 | 0.001653 | 0.002351 |

## Final Conclusion Draft

A random round mintan az LSTM tobb roundban volt jobb MAE szerint.

A lokalis ridge surrogate-ok celja nem uj prediktiv modell tanitasa, hanem az LSTM roundon beluli valoszinuseg-mozgasanak ertelmezheto kozelitese. A suite riportban ezert kulon erdemes kezelni a prediktiv osszehasonlitast es az explainability eredmenyeket.

A utility cohort tabla azt mutatja, hogy az aktiv smoke/inferno, utility damage es recent utility helyzetekben melyik modell valoszinusege volt kozelebb a valos roundkimenethez. Ez lokalis parja a korabbi XGBoost utility ablation elemzesnek, de itt roundon beluli tick-szintu viselkedest mer.
