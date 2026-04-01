import math
from typing import Any, Dict, List
import polars as pl
from awpy import Demo
from src.utils import CSGOUtils

TICKS_PER_SECOND = 64

class DemoProcessor:
    def __init__(self, demo_path: str, tick_step: int = 64):
        self.demo_path = demo_path
        self.tick_step = tick_step
        self.all_map_places = []
        self.site_bounds = {}

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            if isinstance(value, float) and math.isnan(value):
                return default
            return int(value)
        except Exception:
            return default
        
    def _estimate_site_bounds(self, df: pl.DataFrame, site_keywords: List[str], padding: float = 600.0) -> Dict[str, float]:
        pattern = "(?i)" + "|".join(site_keywords)
        site_df = df.filter(pl.col("last_place_name").fill_null("").str.contains(pattern))
        if site_df.height > 0:
            return {
                "x_min": site_df.select(pl.col("X").min()).item() - padding,
                "x_max": site_df.select(pl.col("X").max()).item() + padding,
                "y_min": site_df.select(pl.col("Y").min()).item() - padding,
                "y_max": site_df.select(pl.col("Y").max()).item() + padding,
            }
        return {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0}

    def _site_coordinate_filter(self, site: str) -> pl.Expr:
        bounds = self.site_bounds.get(site)
        if not bounds or bounds["x_max"] == 0.0: return pl.lit(False)
        return ((pl.col("X") >= bounds["x_min"]) & (pl.col("X") <= bounds["x_max"]) & (pl.col("Y") >= bounds["y_min"]) & (pl.col("Y") <= bounds["y_max"]))

    def process(self) -> pl.DataFrame:
        dem = Demo(self.demo_path, verbose=False)
        dem.parse()
        
        player_state_df = dem.parse_ticks([
            "X", "Y", "Z", "health", "is_alive", "armor_value", "has_helmet", "has_defuser",
            "balance", "start_balance", "total_cash_spent", "cash_spent_this_round",
            "round_start_equip_value", "current_equip_value", "inventory", "flash_duration",
            "flash_max_alpha", "molotov_damage_time", "last_place_name", "pitch", "yaw",
            "velocity_X", "velocity_Y", "velocity_Z", "is_scoped", "is_walking",
            "is_defusing", "duck_amount", "shots_fired", "in_bomb_zone", "which_bomb_zone",
            "active_weapon", "steamid", "name", "round_num", "team_num"
        ])

        if player_state_df.height == 0 or dem.rounds.height == 0:
            return pl.DataFrame()

        places_list = player_state_df.select("last_place_name").fill_null("UNKNOWN").unique().to_series().to_list()
        self.all_map_places = [str(p).strip() if p is not None and str(p).strip() else "UNKNOWN" for p in places_list]
        self.site_bounds = {
            "A": self._estimate_site_bounds(player_state_df, ["bombsitea", "sitea", "a site"]),
            "B": self._estimate_site_bounds(player_state_df, ["bombsiteb", "siteb", "b site"])
        }

        player_state_df = self._preprocess_player_state(player_state_df, dem.rounds)
        grenades_df = self._prepare_grenades(dem.grenades, player_state_df)
        smokes_df = dem.smokes.with_columns(pl.col("thrower_side").cast(pl.Utf8).str.to_uppercase())
        infernos_df = dem.infernos.with_columns(pl.col("thrower_side").cast(pl.Utf8).str.to_uppercase())

        return self._build_snapshot_dataset(dem.rounds, player_state_df, smokes_df, infernos_df, grenades_df, dem.kills, dem.damages, dem.bomb)

    def _preprocess_player_state(self, df: pl.DataFrame, rounds_df: pl.DataFrame) -> pl.DataFrame:
        if "side" in df.columns:
            df = df.with_columns(pl.col("side").cast(pl.Utf8).str.to_uppercase().alias("side"))
        else:
            df = df.with_columns(pl.when(pl.col("team_num") == 2).then(pl.lit("T")).when(pl.col("team_num") == 3).then(pl.lit("CT")).otherwise(pl.lit(None)).alias("side"))

        if "round_num" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("round_num"))
            for row in rounds_df.select(["round_num", "start", "end"]).to_dicts():
                start_tick = self._safe_int(row.get("start"), default=-1)
                end_tick = self._safe_int(row.get("end"), default=-1)
                round_num = self._safe_int(row.get("round_num"), default=-1)
                if start_tick < 0 or end_tick < 0 or round_num < 0:
                    continue
                df = df.with_columns(
                    pl.when((pl.col("tick") >= start_tick) & (pl.col("tick") <= end_tick))
                    .then(pl.lit(round_num))
                    .otherwise(pl.col("round_num"))
                    .alias("round_num")
                )
        df = df.filter(pl.col("round_num").is_not_null())

        extra_data = []
        for row in df.select(["inventory", "last_place_name", "velocity_X", "velocity_Y", "velocity_Z"]).to_dicts():
            items = CSGOUtils.normalize_inventory_value(row["inventory"])
            c, l = CSGOUtils.inventory_counts(items), CSGOUtils.extract_loadout(items)
            p_val = row["last_place_name"]
            safe_place = str(p_val).strip() if p_val is not None and str(p_val).strip() else "UNKNOWN"
            extra_data.append({
                "inv_smoke": c["smoke"], "inv_flash": c["flash"], "inv_he": c["he"], "inv_molly": c["molly"], "inv_bomb": c["bomb"], "inv_total": c["total_utility"],
                "primary_weapon": l["primary_weapon"], "secondary_weapon": l["secondary_weapon"],
                "site_bucket": CSGOUtils.generic_site_bucket(safe_place), "safe_place_name": safe_place,
                "velocity_mag": math.sqrt((row["velocity_X"] or 0)**2 + (row["velocity_Y"] or 0)**2 + (row["velocity_Z"] or 0)**2)
            })
        return pl.concat([df, pl.DataFrame(extra_data)], how="horizontal")

    def _prepare_grenades(self, grenades_df: pl.DataFrame, player_state_df: pl.DataFrame) -> pl.DataFrame:
        if grenades_df.height == 0: return grenades_df
        side_map = player_state_df.select(["round_num", "steamid", "side"]).drop_nulls().unique().rename({"steamid": "thrower_steamid", "side": "thrower_side"})
        return grenades_df.join(side_map, on=["round_num", "thrower_steamid"], how="left")

    def _aggregate_team_state(self, team_df: pl.DataFrame, prefix: str) -> Dict[str, Any]:
        base = {f"{prefix}_alive": 0, f"{prefix}_hp_sum": 0.0, f"{prefix}_armor_sum": 0.0, f"{prefix}_money_sum": 0.0, f"{prefix}_start_balance_sum": 0.0, f"{prefix}_cash_spent_round_sum": 0.0, f"{prefix}_equip_value_sum": 0.0, f"{prefix}_round_start_equip_sum": 0.0, f"{prefix}_helmet_count": 0, f"{prefix}_defuser_count": 0, f"{prefix}_scoped_count": 0, f"{prefix}_walking_count": 0, f"{prefix}_defusing_count": 0, f"{prefix}_bomb_zone_count": 0, f"{prefix}_flashed_players": 0, f"{prefix}_flash_duration_sum": 0.0, f"{prefix}_flash_alpha_mean": 0.0, f"{prefix}_burning_players": 0, f"{prefix}_duck_amount_mean": 0.0, f"{prefix}_shots_fired_sum": 0.0, f"{prefix}_velocity_mean": 0.0, f"{prefix}_smoke_inv": 0, f"{prefix}_flash_inv": 0, f"{prefix}_he_inv": 0, f"{prefix}_molly_inv": 0, f"{prefix}_utility_inv": 0, f"{prefix}_bomb_carrier_alive": 0, f"{prefix}_unique_places": 0, f"{prefix}_macro_A": 0, f"{prefix}_macro_B": 0, f"{prefix}_macro_MID": 0, f"{prefix}_macro_OTHER": 0, f"{prefix}_mean_X": 0.0, f"{prefix}_mean_Y": 0.0, f"{prefix}_spread_xy": 0.0}
        for place in self.all_map_places: base[f"{prefix}_place_{place.replace(' ', '_').upper()}"] = 0
        if team_df.height == 0: return base

        agg = team_df.select([
            pl.col("health").sum().alias("hp_sum"), pl.col("armor_value").sum().alias("armor_sum"), pl.col("balance").sum().alias("money_sum"), pl.col("start_balance").sum().alias("start_balance_sum"), pl.col("cash_spent_this_round").sum().alias("cash_spent_round_sum"), pl.col("current_equip_value").sum().alias("equip_value_sum"), pl.col("round_start_equip_value").sum().alias("round_start_equip_sum"), pl.col("has_helmet").cast(pl.Int32).sum().alias("helmet_count"), pl.col("has_defuser").cast(pl.Int32).sum().alias("defuser_count"), pl.col("is_scoped").cast(pl.Int32).sum().alias("scoped_count"), pl.col("is_walking").cast(pl.Int32).sum().alias("walking_count"), pl.col("is_defusing").cast(pl.Int32).sum().alias("defusing_count"), pl.col("in_bomb_zone").cast(pl.Int32).sum().alias("bomb_zone_count"), (pl.col("flash_duration") > 0).cast(pl.Int32).sum().alias("flashed_players"), pl.col("flash_duration").sum().alias("flash_duration_sum"), pl.col("flash_max_alpha").mean().alias("flash_alpha_mean"), (pl.col("molotov_damage_time") > 0).cast(pl.Int32).sum().alias("burning_players"), pl.col("duck_amount").mean().alias("duck_amount_mean"), pl.col("shots_fired").sum().alias("shots_fired_sum"), pl.col("velocity_mag").mean().alias("velocity_mean"), pl.col("inv_smoke").sum().alias("smoke_inv"), pl.col("inv_flash").sum().alias("flash_inv"), pl.col("inv_he").sum().alias("he_inv"), pl.col("inv_molly").sum().alias("molly_inv"), pl.col("inv_total").sum().alias("utility_inv"), pl.col("inv_bomb").sum().alias("bomb_carrier"), pl.col("X").mean().alias("mean_x"), pl.col("Y").mean().alias("mean_y")
        ]).to_dicts()[0]

        unique_p = set()
        macro_counts = {"A": 0, "B": 0, "MID": 0, "OTHER": 0}
        for p in team_df.select(["site_bucket", "safe_place_name"]).to_dicts():
            macro_counts[p["site_bucket"]] += 1
            unique_p.add(p["safe_place_name"])
            base[f"{prefix}_place_{p['safe_place_name'].replace(' ', '_').upper()}"] += 1

        spread = team_df.select((((pl.col("X") - (agg["mean_x"] or 0))**2 + (pl.col("Y") - (agg["mean_y"] or 0))**2)**0.5).mean()).item() if team_df.height > 1 else 0.0

        base.update({
            f"{prefix}_alive": team_df.height, f"{prefix}_hp_sum": agg["hp_sum"] or 0.0, f"{prefix}_armor_sum": agg["armor_sum"] or 0.0, f"{prefix}_money_sum": agg["money_sum"] or 0.0, f"{prefix}_start_balance_sum": agg["start_balance_sum"] or 0.0, f"{prefix}_cash_spent_round_sum": agg["cash_spent_round_sum"] or 0.0, f"{prefix}_equip_value_sum": agg["equip_value_sum"] or 0.0, f"{prefix}_round_start_equip_sum": agg["round_start_equip_sum"] or 0.0, f"{prefix}_helmet_count": agg["helmet_count"] or 0, f"{prefix}_defuser_count": agg["defuser_count"] or 0, f"{prefix}_scoped_count": agg["scoped_count"] or 0, f"{prefix}_walking_count": agg["walking_count"] or 0, f"{prefix}_defusing_count": agg["defusing_count"] or 0, f"{prefix}_bomb_zone_count": agg["bomb_zone_count"] or 0, f"{prefix}_flashed_players": agg["flashed_players"] or 0, f"{prefix}_flash_duration_sum": agg["flash_duration_sum"] or 0.0, f"{prefix}_flash_alpha_mean": agg["flash_alpha_mean"] or 0.0, f"{prefix}_burning_players": agg["burning_players"] or 0, f"{prefix}_duck_amount_mean": agg["duck_amount_mean"] or 0.0, f"{prefix}_shots_fired_sum": agg["shots_fired_sum"] or 0.0, f"{prefix}_velocity_mean": agg["velocity_mean"] or 0.0, f"{prefix}_smoke_inv": agg["smoke_inv"] or 0, f"{prefix}_flash_inv": agg["flash_inv"] or 0, f"{prefix}_he_inv": agg["he_inv"] or 0, f"{prefix}_molly_inv": agg["molly_inv"] or 0, f"{prefix}_utility_inv": agg["utility_inv"] or 0, f"{prefix}_bomb_carrier_alive": 1 if (agg["bomb_carrier"] or 0) > 0 else 0, f"{prefix}_unique_places": len(unique_p), f"{prefix}_macro_A": macro_counts["A"], f"{prefix}_macro_B": macro_counts["B"], f"{prefix}_macro_MID": macro_counts["MID"], f"{prefix}_macro_OTHER": macro_counts["OTHER"], f"{prefix}_mean_X": agg["mean_x"] or 0.0, f"{prefix}_mean_Y": agg["mean_y"] or 0.0, f"{prefix}_spread_xy": spread or 0.0
        })
        return base

    def _build_sum_index(self, df, value_col, side_col="attacker_side"):
        out = {}
        for side in ["T", "CT"]:
            s = df.filter(pl.col(side_col).cast(pl.Utf8).str.to_uppercase() == side).sort("tick")
            ticks = s.select("tick").to_series().to_list()
            vals = s.select(value_col).fill_null(0).to_series().to_list() if s.height > 0 else []
            prefix = [0.0]
            for v in vals: prefix.append(prefix[-1] + float(v))
            out[side] = (ticks, prefix)
        return out

    def _build_count_index(self, df, side_col="attacker_side"):
        return {side: df.filter(pl.col(side_col).cast(pl.Utf8).str.to_uppercase() == side).sort("tick").select("tick").to_series().to_list() for side in ["T", "CT"]}

    def _build_entity_tick_index(self, df):
        out = {}
        if df.height == 0: return out
        tmp = df.group_by("entity_id").agg([pl.col("tick").min().alias("tick"), pl.col("thrower_side").first().cast(pl.Utf8).str.to_uppercase().alias("_side"), pl.col("grenade_type").first().cast(pl.Utf8).str.to_lowercase().alias("_gtype")])
        for row in tmp.group_by(["_side", "_gtype"]).agg(pl.col("tick").sort()).to_dicts(): out[(str(row["_side"]), str(row["_gtype"]))] = [int(x) for x in row["tick"]]
        return out

    def _grenade_count_in_window(self, index_map, side, types, start_tick, end_tick):
        return sum(CSGOUtils.range_count(index_map.get((side.upper(), gtype.lower()), []), start_tick, end_tick) for gtype in types)

    def _count_active_utility_in_site(self, df, tick, side, site):
        if df.height == 0: return 0
        return df.filter((pl.col("start_tick") <= tick) & (pl.col("end_tick") >= tick) & (pl.col("thrower_side") == side) & self._site_coordinate_filter(site)).height

    def _count_active_near_point(self, df, tick, side, x, y, radius=400.0):
        if df.height == 0: return 0
        return df.filter((pl.col("start_tick") <= tick) & (pl.col("end_tick") >= tick) & (pl.col("thrower_side") == side) & (((pl.col("X") - x) ** 2 + (pl.col("Y") - y) ** 2) <= radius ** 2)).height

    def _count_grenade_throws_in_site(self, df, start_tick, end_tick, side, types, site):
        if df.height == 0: return 0
        s = df.filter((pl.col("tick") >= start_tick) & (pl.col("tick") <= end_tick) & (pl.col("thrower_side") == side) & (pl.col("grenade_type").cast(pl.Utf8).str.to_lowercase().is_in([t.lower() for t in types])) & pl.col("X").is_not_null() & pl.col("Y").is_not_null() & self._site_coordinate_filter(site))
        return s.select(pl.col("entity_id").n_unique()).item() if s.height > 0 else 0

    def _build_player_slot(self, p_row, slot_prefix):
        if not p_row: return {f"{slot_prefix}_{k}": v for k, v in {"name": "", "steamid": 0, "alive": 0, "hp": 0.0, "armor": 0.0, "has_helmet": 0, "has_defuser": 0, "is_scoped": 0, "is_walking": 0, "duck_amount": 0.0, "flash_duration": 0.0, "shots_fired": 0.0, "X": 0.0, "Y": 0.0, "place": "MISSING", "primary_weapon": "", "secondary_weapon": "", "smoke": 0, "flash": 0, "he": 0, "molly": 0, "utility_total": 0, "has_bomb": 0}.items()}
        return {
            f"{slot_prefix}_name": str(p_row.get("name") or ""), f"{slot_prefix}_steamid": self._safe_int(p_row.get("steamid"), default=0), f"{slot_prefix}_alive": 1 if (p_row.get("is_alive", False) if "is_alive" in p_row else float(p_row.get("health", 0.0) or 0.0) > 0) else 0, f"{slot_prefix}_hp": float(p_row.get("health") or 0.0), f"{slot_prefix}_armor": float(p_row.get("armor_value") or 0.0), f"{slot_prefix}_has_helmet": 1 if bool(p_row.get("has_helmet", False)) else 0, f"{slot_prefix}_has_defuser": 1 if bool(p_row.get("has_defuser", False)) else 0, f"{slot_prefix}_is_scoped": 1 if bool(p_row.get("is_scoped", False)) else 0, f"{slot_prefix}_is_walking": 1 if bool(p_row.get("is_walking", False)) else 0, f"{slot_prefix}_duck_amount": float(p_row.get("duck_amount") or 0.0), f"{slot_prefix}_flash_duration": float(p_row.get("flash_duration") or 0.0), f"{slot_prefix}_shots_fired": float(p_row.get("shots_fired") or 0.0), f"{slot_prefix}_X": float(p_row.get("X") or 0.0), f"{slot_prefix}_Y": float(p_row.get("Y") or 0.0), f"{slot_prefix}_place": p_row.get("safe_place_name"), f"{slot_prefix}_primary_weapon": p_row.get("primary_weapon"), f"{slot_prefix}_secondary_weapon": p_row.get("secondary_weapon"), f"{slot_prefix}_smoke": p_row.get("inv_smoke"), f"{slot_prefix}_flash": p_row.get("inv_flash"), f"{slot_prefix}_he": p_row.get("inv_he"), f"{slot_prefix}_molly": p_row.get("inv_molly"), f"{slot_prefix}_utility_total": p_row.get("inv_total"), f"{slot_prefix}_has_bomb": 1 if (p_row.get("inv_bomb") or 0) > 0 else 0,
        }

    def _build_snapshot_dataset(self, rounds_df, player_state_df, smokes_df, infernos_df, grenades_df, kills_df, damages_df, bomb_df) -> pl.DataFrame:
        players_by_round = player_state_df.partition_by("round_num", as_dict=True)
        smokes_by_round = smokes_df.partition_by("round_num", as_dict=True) if smokes_df.height > 0 else {}
        infernos_by_round = infernos_df.partition_by("round_num", as_dict=True) if infernos_df.height > 0 else {}
        grenades_by_round = grenades_df.partition_by("round_num", as_dict=True) if grenades_df.height > 0 else {}
        kills_by_round = kills_df.partition_by("round_num", as_dict=True) if kills_df.height > 0 else {}
        damages_by_round = damages_df.partition_by("round_num", as_dict=True) if damages_df.height > 0 else {}
        bomb_by_round = bomb_df.partition_by("round_num", as_dict=True) if bomb_df.height > 0 else {}

        rows = []
        for round_row in rounds_df.to_dicts():
            rnum = self._safe_int(round_row.get("round_num"), default=-1)
            if rnum < 0:
                continue
            key = (rnum,)
            start_tick = self._safe_int(round_row.get("start"), default=-1)
            end_tick = self._safe_int(round_row.get("end"), default=-1)
            if start_tick < 0 or end_tick < 0 or end_tick <= start_tick:
                continue
            freeze_end = self._safe_int(round_row.get("freeze_end"), default=start_tick)
            live_start_tick = max(start_tick, freeze_end)
            if live_start_tick >= end_tick:
                continue
            ct_win = 1 if str(round_row.get("winner", "")).strip().upper() == "CT" else 0

            rp = players_by_round.get(key, pl.DataFrame())
            r_smokes = smokes_by_round.get(key, pl.DataFrame())
            r_infernos = infernos_by_round.get(key, pl.DataFrame())
            r_grenades = grenades_by_round.get(key, pl.DataFrame())
            r_kills = kills_by_round.get(key, pl.DataFrame())
            r_damages = damages_by_round.get(key, pl.DataFrame())
            r_bomb = bomb_by_round.get(key, pl.DataFrame())

            players_by_tick = rp.partition_by("tick", as_dict=True) if rp.height > 0 else {}
            
            t_roster, ct_roster = [], []
            if rp.height > 0:
                t_roster = [int(r["steamid"]) for r in rp.filter((pl.col("side") == "T") & pl.col("steamid").is_not_null()).select(["steamid", "name"]).unique().sort(["name", "steamid"]).head(5).to_dicts()]
                ct_roster = [int(r["steamid"]) for r in rp.filter((pl.col("side") == "CT") & pl.col("steamid").is_not_null()).select(["steamid", "name"]).unique().sort(["name", "steamid"]).head(5).to_dicts()]
            while len(t_roster) < 5: t_roster.append(-len(t_roster) - 1)
            while len(ct_roster) < 5: ct_roster.append(-len(ct_roster) - 101)

            dmg_col = "dmg_health_real" if (r_damages.height > 0 and "dmg_health_real" in r_damages.columns) else "dmg_health"
            if r_damages.height > 0 and dmg_col in r_damages.columns: r_damages = r_damages.with_columns(pl.col(dmg_col).fill_null(0).alias(dmg_col))
            u_damages = r_damages.filter(pl.col("weapon").cast(pl.Utf8).str.to_lowercase().fill_null("").map_elements(lambda w: any(token in w for token in ["hegrenade", "inferno", "molotov", "incendiary"]), return_dtype=pl.Boolean)) if r_damages.height > 0 else pl.DataFrame()

            dmg_idx = self._build_sum_index(r_damages, dmg_col) if r_damages.height > 0 else {"T": ([], [0.0]), "CT": ([], [0.0])}
            udmg_idx = self._build_sum_index(u_damages, dmg_col) if u_damages.height > 0 else {"T": ([], [0.0]), "CT": ([], [0.0])}
            kill_idx = self._build_count_index(r_kills) if r_kills.height > 0 else {"T": [], "CT": []}
            bomb_ticks = r_bomb.sort("tick").select("tick").to_series().to_list() if r_bomb.height > 0 else []
            grenade_idx = self._build_entity_tick_index(r_grenades)

            for tick in range(live_start_tick, end_tick, self.tick_step):
                last_3s, last_5s = max(live_start_tick, tick - 3 * TICKS_PER_SECOND), max(live_start_tick, tick - 5 * TICKS_PER_SECOND)
                
                tick_rows = players_by_tick.get((tick,), rp.head(0))
                if tick_rows.height == 0:
                    continue
                alive_rows = tick_rows.filter(pl.col("is_alive") == True) if "is_alive" in tick_rows.columns else tick_rows.filter(pl.col("health") > 0)
                
                t_alive = alive_rows.filter(pl.col("side") == "T") if alive_rows.height > 0 else pl.DataFrame()
                ct_alive = alive_rows.filter(pl.col("side") == "CT") if alive_rows.height > 0 else pl.DataFrame()

                t_state, ct_state = self._aggregate_team_state(t_alive, "T"), self._aggregate_team_state(ct_alive, "CT")
                
                t_rows, ct_rows = t_alive.select(["X", "Y"]).to_dicts(), ct_alive.select(["X", "Y"]).to_dicts()
                t_mean = sum([min(math.sqrt((float(s["X"]) - float(d["X"])) ** 2 + (float(s["Y"]) - float(d["Y"])) ** 2) for d in ct_rows) for s in t_rows]) / len(t_rows) if t_rows and ct_rows else 0.0
                ct_mean = sum([min(math.sqrt((float(s["X"]) - float(d["X"])) ** 2 + (float(s["Y"]) - float(d["Y"])) ** 2) for d in t_rows) for s in ct_rows]) / len(ct_rows) if ct_rows and t_rows else 0.0

                row = {
                    "round_num": rnum, "tick": tick, "seconds_in_round": (tick - live_start_tick) / TICKS_PER_SECOND,
                    "bomb_planted": 1 if round_row.get("bomb_plant") is not None and tick >= self._safe_int(round_row.get("bomb_plant"), default=10**12) else 0,
                    "bomb_events_last_5s": CSGOUtils.range_count(bomb_ticks, last_5s, tick),
                    **t_state, **ct_state,
                    "alive_diff": ct_state["CT_alive"] - t_state["T_alive"], "hp_diff": ct_state["CT_hp_sum"] - t_state["T_hp_sum"],
                    "armor_diff": ct_state["CT_armor_sum"] - t_state["T_armor_sum"], "money_diff": ct_state["CT_money_sum"] - t_state["T_money_sum"],
                    "equip_diff": ct_state["CT_equip_value_sum"] - t_state["T_equip_value_sum"], "utility_inv_diff": ct_state["CT_utility_inv"] - t_state["T_utility_inv"],
                    "flash_inv_diff": ct_state["CT_flash_inv"] - t_state["T_flash_inv"], "smoke_inv_diff": ct_state["CT_smoke_inv"] - t_state["T_smoke_inv"],
                    "molly_inv_diff": ct_state["CT_molly_inv"] - t_state["T_molly_inv"], "spread_diff": ct_state["CT_spread_xy"] - t_state["T_spread_xy"],
                    "centroid_distance_xy": math.sqrt((ct_state["CT_mean_X"] - t_state["T_mean_X"]) ** 2 + (ct_state["CT_mean_Y"] - t_state["T_mean_Y"]) ** 2),
                    "T_closest_enemy_dist": t_mean, "CT_closest_enemy_dist": ct_mean, "closest_enemy_dist_diff": ct_mean - t_mean,
                    "T_damage_last_5s": CSGOUtils.range_sum(dmg_idx["T"][0], dmg_idx["T"][1], last_5s, tick),
                    "CT_damage_last_5s": CSGOUtils.range_sum(dmg_idx["CT"][0], dmg_idx["CT"][1], last_5s, tick),
                    "T_utility_damage_last_5s": CSGOUtils.range_sum(udmg_idx["T"][0], udmg_idx["T"][1], last_5s, tick),
                    "CT_utility_damage_last_5s": CSGOUtils.range_sum(udmg_idx["CT"][0], udmg_idx["CT"][1], last_5s, tick),
                    "T_kills_last_3s": CSGOUtils.range_count(kill_idx["T"], last_3s, tick),
                    "CT_kills_last_3s": CSGOUtils.range_count(kill_idx["CT"], last_3s, tick),
                    "T_smokes_last_5s": self._grenade_count_in_window(grenade_idx, "T", ["CSmokeGrenade"], last_5s, tick),
                    "CT_smokes_last_5s": self._grenade_count_in_window(grenade_idx, "CT", ["CSmokeGrenade"], last_5s, tick),
                    "T_flashes_last_5s": self._grenade_count_in_window(grenade_idx, "T", ["CFlashbang"], last_5s, tick),
                    "CT_flashes_last_5s": self._grenade_count_in_window(grenade_idx, "CT", ["CFlashbang"], last_5s, tick),
                    "T_he_last_5s": self._grenade_count_in_window(grenade_idx, "T", ["CHEGrenade"], last_5s, tick),
                    "CT_he_last_5s": self._grenade_count_in_window(grenade_idx, "CT", ["CHEGrenade"], last_5s, tick),
                    "T_mollies_last_5s": self._grenade_count_in_window(grenade_idx, "T", ["CMolotovGrenade", "CIncendiaryGrenade"], last_5s, tick),
                    "CT_mollies_last_5s": self._grenade_count_in_window(grenade_idx, "CT", ["CMolotovGrenade", "CIncendiaryGrenade"], last_5s, tick),
                    **{f"T{idx+1}_{k}": v for idx, pid in enumerate(t_roster) for k, v in self._build_player_slot({self._safe_int(r.get("steamid"), default=0): r for r in tick_rows.to_dicts() if r.get("steamid") is not None}.get(self._safe_int(pid, default=0), {}), "").items()},
                    **{f"CT{idx+1}_{k}": v for idx, pid in enumerate(ct_roster) for k, v in self._build_player_slot({self._safe_int(r.get("steamid"), default=0): r for r in tick_rows.to_dicts() if r.get("steamid") is not None}.get(self._safe_int(pid, default=0), {}), "").items()},
                }
                
                row["damage_diff_last_5s"] = row["CT_damage_last_5s"] - row["T_damage_last_5s"]
                row["utility_damage_diff_last_5s"] = row["CT_utility_damage_last_5s"] - row["T_utility_damage_last_5s"]
                row["kill_diff_last_3s"] = row["CT_kills_last_3s"] - row["T_kills_last_3s"]

                for site in ["A", "B"]:
                    for side in ["T", "CT"]:
                        pfx = f"{side}_{site}_site"
                        row.update({
                            f"{pfx}_active_smokes": self._count_active_utility_in_site(r_smokes, tick, side, site),
                            f"{pfx}_active_infernos": self._count_active_utility_in_site(r_infernos, tick, side, site),
                            f"{pfx}_smokes_last_5s": self._count_grenade_throws_in_site(r_grenades, last_5s, tick, side, ["CSmokeGrenade"], site),
                            f"{pfx}_flashes_last_5s": self._count_grenade_throws_in_site(r_grenades, last_5s, tick, side, ["CFlashbang"], site),
                            f"{pfx}_he_last_5s": self._count_grenade_throws_in_site(r_grenades, last_5s, tick, side, ["CHEGrenade"], site),
                            f"{pfx}_mollies_last_5s": self._count_grenade_throws_in_site(r_grenades, last_5s, tick, side, ["CMolotovGrenade", "CIncendiaryGrenade"], site),
                        })
                        
                row["T_active_smokes"] = r_smokes.filter((pl.col("start_tick") <= tick) & (pl.col("end_tick") >= tick) & (pl.col("thrower_side") == "T")).height if r_smokes.height > 0 else 0
                row["CT_active_smokes"] = r_smokes.filter((pl.col("start_tick") <= tick) & (pl.col("end_tick") >= tick) & (pl.col("thrower_side") == "CT")).height if r_smokes.height > 0 else 0
                row["T_active_infernos"] = r_infernos.filter((pl.col("start_tick") <= tick) & (pl.col("end_tick") >= tick) & (pl.col("thrower_side") == "T")).height if r_infernos.height > 0 else 0
                row["CT_active_infernos"] = r_infernos.filter((pl.col("start_tick") <= tick) & (pl.col("end_tick") >= tick) & (pl.col("thrower_side") == "CT")).height if r_infernos.height > 0 else 0
                row["active_smokes_total"] = row["T_active_smokes"] + row["CT_active_smokes"]
                row["active_infernos_total"] = row["T_active_infernos"] + row["CT_active_infernos"]
                row["ct_win"] = ct_win

                rows.append(row)

        return pl.DataFrame(rows)
