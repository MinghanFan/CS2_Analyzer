from awpy import Demo
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# ===== Configuration =====
demo_root = Path("/Volumes/TOSHIBA EXT/Demo_2025")  # Change to your root directory
output_csv = "weapon_duel_economy_analysis.csv"
EQUAL_THRESHOLD = 200  # ±$200 for equal economy
DEFAULT_TICKRATE = 64

# ===== Weapon Price Dictionary =====
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

# ===== Valid weapons for knife round detection =====
valid_guns = {
    "hkp2000", "elite", "glock", "p250", "fiveseven", "tec9", "cz75a", "deagle", "revolver",
    "mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon",
    "galilar", "famas", "ak47", "m4a1", "m4a1_silencer", "sg556", "aug",
    "ssg08", "awp", "g3sg1", "scar20",
    "nova", "xm1014", "mag7", "sawedoff",
    "negev", "m249", "taser"
}

# ===== Name normalization =====
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

def get_weapon_value(weapon_name):
    """Get weapon value from price dict, return 0 for knife/grenades/unknown"""
    if pd.isna(weapon_name) or not isinstance(weapon_name, str):
        return 0
    
    weapon_name = weapon_name.strip()
    
    # Knife and grenades = $0
    knife_grenade_keywords = ["knife", "grenade", "molotov", "incendiary", "flashbang", 
                              "smoke", "decoy", "c4"]
    if any(kw in weapon_name.lower() for kw in knife_grenade_keywords):
        return 0
    
    # Look up in price dictionary
    return weapon_prices.get(weapon_name, 0)

# ===== Find demo files =====
demo_files = [f for f in sorted(demo_root.rglob("*.dem")) if not f.name.startswith("._")]
print(f"Found {len(demo_files)} demos under {demo_root}")

# ===== Global aggregators =====
# player -> awp_filter -> category -> count
# awp_filter: "include_awp" or "exclude_awp"
# categories: kills/deaths by economy condition
player_stats = defaultdict(lambda: {
    "include_awp": {
        "higher_econ_kills": 0,
        "equal_econ_kills": 0,
        "lower_econ_kills": 0,
        "higher_econ_deaths": 0,
        "equal_econ_deaths": 0,
        "lower_econ_deaths": 0,
        "total_kills": 0,
        "total_deaths": 0
    },
    "exclude_awp": {
        "higher_econ_kills": 0,
        "equal_econ_kills": 0,
        "lower_econ_kills": 0,
        "higher_econ_deaths": 0,
        "equal_econ_deaths": 0,
        "lower_econ_deaths": 0,
        "total_kills": 0,
        "total_deaths": 0
    },
    "rounds_participated": 0
})

countknife = 0

for demo_path in demo_files:
    print(f"\nParsing {demo_path.name}")
    try:
        demo = Demo(str(demo_path), verbose=False)
        demo.parse(player_props=["name", "side"])
        
        rounds_df = demo.rounds.to_pandas()
        ticks_df = demo.ticks.to_pandas()
        kills_df = demo.kills.to_pandas()
        damages_df = demo.damages.to_pandas()
    except Exception as e:
        print(f"[warn] failed on {demo_path.name}: {e}")
        continue

    if rounds_df is None or rounds_df.empty:
        print(f"[warn] {demo_path.name} has empty rounds data")
        continue

    # ===== Remove knife/warmup round =====
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
                if kills_df is not None and not kills_df.empty:
                    kills_df = kills_df[kills_df["round_num"] != first_round]

    # ===== Track round participation from ticks (like econ_adv.py) =====
    if ticks_df is not None and not ticks_df.empty and "name" in ticks_df.columns and "round_num" in ticks_df.columns:
        ticks_df_clean = ticks_df.dropna(subset=["name", "round_num"]).copy()
        ticks_df_clean["name"] = ticks_df_clean["name"].apply(lambda x: norm_name(str(x)))
        
        # For each round, find unique players
        for round_num in rounds_df["round_num"].unique():
            round_ticks = ticks_df_clean[ticks_df_clean["round_num"] == round_num]
            if not round_ticks.empty:
                players_in_round = round_ticks["name"].unique()
                for player in players_in_round:
                    player_stats[player]["rounds_participated"] += 1

    # ===== Process kills =====
    if kills_df is None or kills_df.empty:
        print(f"[info] {demo_path.name} has no kills data")
        continue
    
    # Check for required columns
    required_cols = ["attacker_name", "attacker_side", "victim_name", "victim_side", 
                     "attacker_active_weapon_name", "victim_active_weapon_name"]
    missing = [c for c in required_cols if c not in kills_df.columns]
    
    if missing:
        print(f"[warn] {demo_path.name} kills missing columns: {missing}")
        continue
    
    # Clean kills data
    kills_df = kills_df.dropna(subset=["attacker_name", "victim_name", "attacker_side", "victim_side"])
    
    # Normalize names
    kills_df["attacker_name"] = kills_df["attacker_name"].apply(norm_name)
    kills_df["victim_name"] = kills_df["victim_name"].apply(norm_name)
    
    # Exclude teamkills
    kills_df = kills_df[kills_df["attacker_side"] != kills_df["victim_side"]]
    
    # Process each kill
    for _, kill in kills_df.iterrows():
        attacker = kill["attacker_name"]
        victim = kill["victim_name"]
        attacker_weapon = kill["attacker_active_weapon_name"]
        victim_weapon = kill["victim_active_weapon_name"]
        
        # Get weapon values
        attacker_value = get_weapon_value(attacker_weapon)
        victim_value = get_weapon_value(victim_weapon)
        
        # Determine if AWP is involved
        is_awp_duel = (str(attacker_weapon).strip() == "AWP" or 
                       str(victim_weapon).strip() == "AWP")
        
        # Determine economy condition for this kill
        value_diff = attacker_value - victim_value
        
        if value_diff > EQUAL_THRESHOLD:
            # Attacker had higher economy
            kill_category = "higher_econ_kills"
            death_category = "lower_econ_deaths"
        elif value_diff < -EQUAL_THRESHOLD:
            # Attacker had lower economy
            kill_category = "lower_econ_kills"
            death_category = "higher_econ_deaths"
        else:
            # Equal economy
            kill_category = "equal_econ_kills"
            death_category = "equal_econ_deaths"
        
        # Update stats for INCLUDE_AWP version (always)
        player_stats[attacker]["include_awp"][kill_category] += 1
        player_stats[attacker]["include_awp"]["total_kills"] += 1
        
        player_stats[victim]["include_awp"][death_category] += 1
        player_stats[victim]["include_awp"]["total_deaths"] += 1
        
        # Update stats for EXCLUDE_AWP version (only if no AWP involved)
        if not is_awp_duel:
            player_stats[attacker]["exclude_awp"][kill_category] += 1
            player_stats[attacker]["exclude_awp"]["total_kills"] += 1
            
            player_stats[victim]["exclude_awp"][death_category] += 1
            player_stats[victim]["exclude_awp"]["total_deaths"] += 1

print(f"\n{'='*70}")
print("GENERATING OUTPUT CSV")
print(f"{'='*70}")

# ===== Generate output CSV =====
results = []

for player, data in player_stats.items():
    rounds_participated = data["rounds_participated"]
    
    # Generate two rows per player: include_awp and exclude_awp
    for awp_filter in ["include_awp", "exclude_awp"]:
        stats = data[awp_filter]
        
        results.append({
            "Player": player,
            "AWP_Filter": awp_filter,
            "Total_Kills": stats["total_kills"],
            "Higher_Econ_Kills": stats["higher_econ_kills"],
            "Equal_Econ_Kills": stats["equal_econ_kills"],
            "Lower_Econ_Kills": stats["lower_econ_kills"],
            "Total_Deaths": stats["total_deaths"],
            "Higher_Econ_Deaths": stats["higher_econ_deaths"],
            "Equal_Econ_Deaths": stats["equal_econ_deaths"],
            "Lower_Econ_Deaths": stats["lower_econ_deaths"],
            "Total_Rounds": rounds_participated
        })

# Create DataFrame and save
df = pd.DataFrame(results)
df = df.sort_values(["Player", "AWP_Filter"])
df.to_csv(output_csv, index=False)

print(f"\nDone! Results saved to {output_csv}")
print(f"Knife rounds removed: {countknife}")
print(f"Total players tracked: {len(player_stats)}")

# ===== Print summary statistics =====
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

if not df.empty:
    # Stats for include_awp
    include_df = df[df["AWP_Filter"] == "include_awp"]
    exclude_df = df[df["AWP_Filter"] == "exclude_awp"]
    
    print("\n--- INCLUDING AWP DUELS ---")
    total_kills_inc = include_df["Total_Kills"].sum()
    higher_kills_inc = include_df["Higher_Econ_Kills"].sum()
    equal_kills_inc = include_df["Equal_Econ_Kills"].sum()
    lower_kills_inc = include_df["Lower_Econ_Kills"].sum()
    
    print(f"Total kills: {total_kills_inc}")
    print(f"  Higher economy kills: {higher_kills_inc} ({higher_kills_inc/total_kills_inc*100:.1f}%)")
    print(f"  Equal economy kills: {equal_kills_inc} ({equal_kills_inc/total_kills_inc*100:.1f}%)")
    print(f"  Lower economy kills: {lower_kills_inc} ({lower_kills_inc/total_kills_inc*100:.1f}%)")
    
    print("\n--- EXCLUDING AWP DUELS ---")
    total_kills_exc = exclude_df["Total_Kills"].sum()
    higher_kills_exc = exclude_df["Higher_Econ_Kills"].sum()
    equal_kills_exc = exclude_df["Equal_Econ_Kills"].sum()
    lower_kills_exc = exclude_df["Lower_Econ_Kills"].sum()
    
    print(f"Total kills: {total_kills_exc}")
    print(f"  Higher economy kills: {higher_kills_exc} ({higher_kills_exc/total_kills_exc*100:.1f}%)")
    print(f"  Equal economy kills: {equal_kills_exc} ({equal_kills_exc/total_kills_exc*100:.1f}%)")
    print(f"  Lower economy kills: {lower_kills_exc} ({lower_kills_exc/total_kills_exc*100:.1f}%)")
    
    print(f"\nAWP duels removed: {total_kills_inc - total_kills_exc} ({(total_kills_inc-total_kills_exc)/total_kills_inc*100:.1f}%)")
    
    # Top performers in lower economy kills (exclude_awp version)
    print("\n" + "="*70)
    print("TOP 10 LOWER ECONOMY KILLERS (EXCLUDING AWP, 50+ ROUNDS)")
    print("="*70)
    
    significant = exclude_df[exclude_df["Total_Rounds"] >= 50].copy()
    if not significant.empty:
        significant["Lower_Econ_Kill_Rate"] = (
            significant["Lower_Econ_Kills"] / significant["Total_Kills"] * 100
        )
        top_lower = significant.nlargest(10, "Lower_Econ_Kill_Rate")
        
        for _, row in top_lower.iterrows():
            print(f"{row['Player']:.<25} {row['Lower_Econ_Kill_Rate']:>5.1f}% "
                  f"({row['Lower_Econ_Kills']}/{row['Total_Kills']} kills, "
                  f"{row['Total_Rounds']} rounds)")