from awpy import Demo
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# === Weapon Price Dictionary ===
weapon_prices = {
    # Pistols
    "Glock-18": 200, "USP-S": 200, "P250": 300, "P2000": 200, "Five-SeveN": 500,
    "Tec-9": 500, "Dual Berettas": 300, "Desert Eagle": 700, "CZ75-Auto": 500, "R8 Revolver": 600,
    # SMGs
    "MAC-10": 1050, "MP9": 1250, "MP7": 1500, "MP5-SD": 1500, "UMP-45": 1200, "P90": 2350, "PP-Bizon": 1400,
    # Rifles
    "Galil AR": 1800, "FAMAS": 1950, "AK-47": 2700, "M4A1-S": 2900,
    "M4A4": 2900, "SG 553": 3000, "AUG": 3300,
    # Sniper Rifles
    "SSG 08": 1700, "AWP": 4750, "G3SG1": 5000, "SCAR-20": 5000,
    # Shotguns
    "Nova": 1050, "XM1014": 2000, "MAG-7": 1300, "Sawed-Off": 1100,
    # Machine Guns
    "Negev": 1700, "M249": 5200,
    # Misc
    "Zeus x27": 200
}

# === Valid weapons for knife round detection ===
valid_guns = {
    "hkp2000", "elite", "glock", "p250", "fiveseven", "tec9", "cz75a", "deagle", "revolver", "usp_silencer",
    "mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon",
    "galilar", "famas", "ak47", "m4a1", "m4a1_silencer", "sg556", "aug",
    "ssg08", "awp", "g3sg1", "scar20",
    "nova", "xm1014", "mag7", "sawedoff",
    "negev", "m249", "taser"
}

# === Name normalization ===
alias_map = {
    "sh1ro": {"SH1R0", "sh1r0"},
    "910": {"910-", "-910"},
    "mzinho": {"Mzinho"},
    "Techno": {"Techno4K"},
    "Ag1l": {"ag1l", "ag1L"},
    "dav1deuS": {"dav1deu$", "davideuS"},
    "device": {"dev1ce"},
    "electroNic": {"electronic"},
    "HeavyGod": {"HeavyGoD"},
    "hfah": {"Hfah"},
    "huNter-": {"huNter"},
    "hypex": {"Hypex"},
    "jcobbb": {"Jcobbb"},
    "kauez": {"Kauez"},
    "lux": {"Lux"},
    "NAF": {"NAF-FLY"},
    "NertZ": {"nertZ"},
    "skullz": {"Skullz"},
    "Snax": {"snax"},
    "woxic": {"Woxic"},
}

def norm_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    for canon, vars_ in alias_map.items():
        if s == canon or s in vars_:
            return canon
    return s

def calc_weapon_value(inv):
    if isinstance(inv, (list, np.ndarray)):
        return sum(weapon_prices.get(item, 0) for item in inv)
    return 0

# === Configuration ===
demo_root = Path("/Volumes/TOSHIBA EXT/Demo_2025")  # Root directory containing event folders
output_csv = "weapon_economy_percentage.csv"

# === Global aggregators ===
# player -> event -> {total_weapon_value, total_percentage, rounds_played}
player_event_stats = defaultdict(lambda: defaultdict(lambda: {
    "total_weapon_value": 0,
    "total_percentage": 0.0,
    "rounds_played": 0
}))

countknife = 0

# === Find all event folders ===
event_folders = [f for f in demo_root.iterdir() if f.is_dir() and not f.name.startswith(".")]
print(f"Found {len(event_folders)} event folders under {demo_root}")

for event_folder in sorted(event_folders):
    event_name = event_folder.name
    print(f"\n{'='*70}")
    print(f"Processing Event: {event_name}")
    print(f"{'='*70}")
    
    # Find demo files in this event folder
    demo_files = [f for f in event_folder.rglob("*.dem") if not f.name.startswith("._")]
    print(f"Found {len(demo_files)} demos in {event_name}")
    
    for demo_path in demo_files:
        print(f"\nParsing {demo_path.name}")
        try:
            demo = Demo(str(demo_path), verbose=False)
            demo.parse(player_props=["name", "inventory", "side"])
            
            rounds_df = demo.rounds.to_pandas()
            ticks_df = demo.ticks.to_pandas()
            damages_df = demo.damages.to_pandas()
        except Exception as e:
            print(f"[warn] failed on {demo_path.name}: {e}")
            continue

        if rounds_df is None or rounds_df.empty:
            print(f"[warn] {demo_path.name} has empty rounds data")
            continue

        # === Remove knife/warmup round ===
        if not rounds_df.empty and damages_df is not None and not damages_df.empty:
            first_round = int(rounds_df["round_num"].min())
            first_round_damages = damages_df[damages_df["round_num"] == first_round]
            
            if not first_round_damages.empty and "weapon" in first_round_damages.columns:
                used_weapons = set(first_round_damages["weapon"].dropna().astype(str).str.lower().unique())
                
                if used_weapons.isdisjoint(valid_guns):
                    print(f"[INFO] Removing knife round {first_round} from {demo_path.name}")
                    countknife += 1
                    rounds_df = rounds_df[rounds_df["round_num"] != first_round]
                    if ticks_df is not None and not ticks_df.empty:
                        ticks_df = ticks_df[ticks_df["round_num"] != first_round]

        # === Process each round ===
        for _, rnd in rounds_df.iterrows():
            round_num = rnd["round_num"]
            freeze_end = rnd["freeze_end"]
            
            if pd.isna(round_num) or pd.isna(freeze_end):
                continue
            
            round_num = int(round_num)
            freeze_end = int(freeze_end)
            
            # Get ticks from freeze_end to freeze_end + 16
            tick_slice = ticks_df[
                (ticks_df["tick"] >= freeze_end) & 
                (ticks_df["tick"] <= freeze_end + 16) &
                (ticks_df["round_num"] == round_num)
            ].dropna(subset=["name", "inventory", "side"])
            
            if tick_slice.empty:
                continue
            
            # Calculate weapon value for each tick
            tick_slice = tick_slice.copy()
            tick_slice["weapon_value"] = tick_slice["inventory"].apply(calc_weapon_value)
            
            # Get most common weapon value for each player
            grouped = tick_slice.groupby(["name", "side"])["weapon_value"].agg(
                lambda x: x.value_counts().idxmax() if not x.empty else 0
            ).reset_index()
            
            # Normalize names
            grouped["name"] = grouped["name"].apply(norm_name)
            
            # Calculate team totals for each side
            team_totals = grouped.groupby("side")["weapon_value"].sum().to_dict()
            
            # Calculate percentage for each player
            for _, row in grouped.iterrows():
                player = row["name"]
                side = row["side"]
                weapon_value = row["weapon_value"]
                
                team_total = team_totals.get(side, 0)
                percentage = (weapon_value / team_total * 100) if team_total > 0 else 0
                
                # Update player stats for this event
                player_event_stats[player][event_name]["total_weapon_value"] += weapon_value
                player_event_stats[player][event_name]["total_percentage"] += percentage
                player_event_stats[player][event_name]["rounds_played"] += 1

print(f"\n{'='*70}")
print("GENERATING OUTPUT CSV")
print(f"{'='*70}")

# === Generate output CSV ===
results = []

for player, events in player_event_stats.items():
    # Add per-event stats
    for event_name, stats in events.items():
        rounds_played = stats["rounds_played"]
        avg_weapon_value = stats["total_weapon_value"] / rounds_played if rounds_played > 0 else 0
        avg_percentage = stats["total_percentage"] / rounds_played if rounds_played > 0 else 0
        
        results.append({
            "Player": player,
            "Event": event_name,
            "RoundsPlayed": rounds_played,
            "AvgWeaponValue": round(avg_weapon_value, 2),
            "AvgPercentageOfTeam": round(avg_percentage, 2)
        })
    
    # Add overall stats
    total_rounds = sum(stats["rounds_played"] for stats in events.values())
    total_weapon_value = sum(stats["total_weapon_value"] for stats in events.values())
    total_percentage = sum(stats["total_percentage"] for stats in events.values())
    
    overall_avg_weapon_value = total_weapon_value / total_rounds if total_rounds > 0 else 0
    overall_avg_percentage = total_percentage / total_rounds if total_rounds > 0 else 0
    
    results.append({
        "Player": player,
        "Event": "overall",
        "RoundsPlayed": total_rounds,
        "AvgWeaponValue": round(overall_avg_weapon_value, 2),
        "AvgPercentageOfTeam": round(overall_avg_percentage, 2)
    })

# Create DataFrame and save
df = pd.DataFrame(results)
df = df.sort_values(["Player", "Event"])
df.to_csv(output_csv, index=False)

print(f"\nDone! Results saved to {output_csv}")
print(f"Knife rounds removed: {countknife}")
print(f"Total players tracked: {len(player_event_stats)}")
print(f"Total rows in CSV: {len(df)}")