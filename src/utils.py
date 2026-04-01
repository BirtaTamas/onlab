import ast
import re
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List

class CSGOUtils:
    MAP_NAME_ALIASES = {
        "anubis": ["anubis", "de_anubis"],
        "ancient": ["ancient", "de_ancient"],
        "assault": ["assault", "cs_assault"],
        "cache": ["cache", "de_cache"],
        "canals": ["canals", "de_canals"],
        "cobblestone": ["cobblestone", "cbble", "de_cbble"],
        "dust2": ["dust2", "dust_2", "de_dust2", "de_dust_2"],
        "inferno": ["inferno", "de_inferno"],
        "italy": ["italy", "cs_italy"],
        "militia": ["militia", "cs_militia"],
        "mirage": ["mirage", "de_mirage"],
        "nuke": ["nuke", "de_nuke"],
        "office": ["office", "cs_office"],
        "overpass": ["overpass", "de_overpass"],
        "train": ["train", "de_train"],
        "vertigo": ["vertigo", "de_vertigo"],
    }

    SECONDARY_WEAPON_TOKENS = [
        "glock", "usp", "p2000", "p250", "fiveseven", "five-seven", 
        "tec-9", "tec9", "cz75", "dual berettas", "elite", "deagle", "r8 revolver"
    ]

    @staticmethod
    def normalize_inventory_value(inv: Any) -> List[str]:
        if inv is None: return []
        if isinstance(inv, list): return [str(x).lower() for x in inv]
        raw = str(inv).strip()
        if raw == "" or raw.lower() in {"none", "null", "nan"}: return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list): return [str(x).lower() for x in parsed]
        except Exception: pass
        return [part.strip().lower() for part in raw.split(",") if part.strip()]

    @staticmethod
    def inventory_counts(items: List[str]) -> Dict[str, int]:
        counts = {"smoke": 0, "flash": 0, "he": 0, "molly": 0, "bomb": 0}
        for item in items:
            if "smoke" in item: counts["smoke"] += 1
            if "flash" in item: counts["flash"] += 1
            if "hegrenade" in item or item.endswith("he"): counts["he"] += 1
            if "molotov" in item or "incendiary" in item: counts["molly"] += 1
            if "c4" in item or item == "bomb": counts["bomb"] += 1
        counts["total_utility"] = counts["smoke"] + counts["flash"] + counts["he"] + counts["molly"]
        return counts

    @staticmethod
    def extract_loadout(items: List[str]) -> Dict[str, str]:
        primary, secondary = "", ""
        for item in items:
            if any(token in item for token in ["flash", "smoke", "molotov", "incendiary", "hegrenade", "decoy", "knife", "bayonet", "karambit", "dagger", "c4", "bomb", "zeus"]):
                continue
            is_sec = any(token in item for token in CSGOUtils.SECONDARY_WEAPON_TOKENS)
            if is_sec and secondary == "": secondary = item
            elif not is_sec and primary == "": primary = item
        return {"primary_weapon": primary, "secondary_weapon": secondary}

    @staticmethod
    def generic_site_bucket(place: str) -> str:
        text = str(place or "").lower().replace(" ", "")
        if any(k in text for k in ["bombsitea", "sitea"]): return "A"
        if any(k in text for k in ["bombsiteb", "siteb"]): return "B"
        if any(k in text for k in ["mid", "middle"]): return "MID"
        return "OTHER"

    @staticmethod
    def range_sum(ticks: List[int], prefix: List[float], start_tick: int, end_tick: int) -> float:
        left, right = bisect_left(ticks, start_tick), bisect_right(ticks, end_tick)
        return float(prefix[right] - prefix[left]) if left < len(prefix) and right < len(prefix) else 0.0

    @staticmethod
    def range_count(ticks: List[int], start_tick: int, end_tick: int) -> int:
        return int(bisect_right(ticks, end_tick) - bisect_left(ticks, start_tick))

    @staticmethod
    def infer_map_name(*texts: str) -> str:
        normalized_text = " ".join(str(t or "").lower() for t in texts)

        for canonical, aliases in CSGOUtils.MAP_NAME_ALIASES.items():
            for alias in aliases:
                pattern = rf"(^|[^a-z0-9]){re.escape(alias.lower())}([^a-z0-9]|$)"
                if re.search(pattern, normalized_text):
                    return canonical

        return "match"
