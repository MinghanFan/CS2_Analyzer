from awpy import Demo
from pathlib import Path
import pandas as pd
from collections import defaultdict

# ===== Configuration =====
demo_root = Path("/Volumes/TOSHIBA EXT/Demo_2025/IEM_Chengdu_2025")  # Change to your root directory
output_csv = "player_kills_verification.csv"
REMOVE_KNIFE_ROUND = True  # Set to True to remove knife rounds

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

# ===== Find demo files =====
demo_files = [f for f in sorted(demo_root.rglob("*.dem")) if not f.name.startswith("._")]
print(f"Found {len(demo_files)} demos under {demo_root}")
print(f"Remove knife rounds: {REMOVE_KNIFE_ROUND}")

# ===== Aggregator =====
player_kills = defaultdict(int)

for demo_path in demo_files:
    print(f"Parsing {demo_path.name}")
    try:
        demo = Demo(str(demo_path), verbose=False)
        demo.parse()
        
        kills_df = demo.kills.to_pandas()
        
        if REMOVE_KNIFE_ROUND:
            rounds_df = demo.rounds.to_pandas()
            damages_df = demo.damages.to_pandas()
    except Exception as e:
        print(f"[warn] failed on {demo_path.name}: {e}")
        continue

    if kills_df is None or kills_df.empty:
        print(f"[warn] {demo_path.name} has no kills data")
        continue
    
    # ===== Remove knife/warmup round if flag is set =====
    if REMOVE_KNIFE_ROUND:
        # if rounds_df is not None and not rounds_df.empty and damages_df is not None and not damages_df.empty:
        #     first_round = int(rounds_df["round_num"].min())
        #     first_round_damages = damages_df[damages_df["round_num"] == first_round]
            
        #     if not first_round_damages.empty and "weapon" in first_round_damages.columns:
        #         used_weapons = set(first_round_damages["weapon"].dropna().astype(str).str.lower().unique())
                
        #         if used_weapons and used_weapons.isdisjoint(valid_guns):
        #             print(f"[INFO] Removing knife round {first_round} from {demo_path.name}")
        #             kills_df = kills_df[kills_df["round_num"] != first_round]
        if not rounds_df.empty and not damages_df.empty:
            first_round = int(rounds_df["round_num"].min())
            first_round_damages = damages_df[damages_df["round_num"] == first_round]
            used_weapons = set(first_round_damages["weapon"].dropna().astype(str).str.lower().unique())
            print(f"[DEBUG] Demo {demo_path.name} round {first_round} used weapons: {used_weapons}")

            if used_weapons.isdisjoint(valid_guns): 
                print(f"[INFO] 剔除 demo {demo_path.name} 的第一个回合 (round {first_round})，武器={used_weapons}")
                rounds_df = rounds_df[rounds_df["round_num"] != first_round]
                damages_df = damages_df[damages_df["round_num"] != first_round]
                # ticks_df = ticks_df[ticks_df["round_num"] != first_round]
                kills_df = kills_df[kills_df["round_num"] != first_round]
    
    # Count kills per player
    if "attacker_name" not in kills_df.columns:
        print(f"[warn] {demo_path.name} kills missing attacker_name")
        continue
    
    kills_clean = kills_df.dropna(subset=["attacker_name"]).copy()
    kills_clean["attacker_name"] = kills_clean["attacker_name"].apply(lambda x: norm_name(str(x)))
    
    # Aggregate
    kill_counts = kills_clean["attacker_name"].value_counts()
    for player, count in kill_counts.items():
        player_kills[player] += count

# ===== Generate output =====
if player_kills:
    rows = [{"Player": player, "TotalKills": kills} for player, kills in player_kills.items()]
    df = pd.DataFrame(rows)
    df = df.sort_values("TotalKills", ascending=False)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*50}")
    print(f"Results saved to {output_csv}")
    print(f"{'='*50}")
    print(f"Total players: {len(df)}")
    print(f"Total kills: {df['TotalKills'].sum()}")
    print(f"\nTop 10 players by kills:")
    print(df.head(10).to_string(index=False))
else:
    print("No kills data found.")