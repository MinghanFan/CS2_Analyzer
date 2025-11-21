from awpy import Demo
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# ===== Configuration =====
demo_root = Path("/Volumes/TOSHIBA EXT/Demo_2025")  # Change to your root directory
output_csv = "exit_frag_analysis.csv"
DEFAULT_TICKRATE = 64

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

def get_tickrate_from_header(demo, default=64):
    """Extract tickrate from demo header"""
    try:
        hd = demo.header.to_pandas() if hasattr(demo.header, "to_pandas") else pd.DataFrame()
        if not hd.empty:
            for col in ["tick_rate", "tickrate", "tickRate"]:
                if col in hd.columns and pd.notna(hd.loc[0, col]):
                    return int(hd.loc[0, col])
    except Exception as e:
        print(f"[warn] tickrate read failed: {e}")
    return default

# ===== Find demo files =====
demo_files = [f for f in sorted(demo_root.rglob("*.dem")) if not f.name.startswith("._")]
print(f"Found {len(demo_files)} demos under {demo_root}")

# ===== Global aggregators =====
# player -> {"total_kills": int, "exit_frags": int, "meaningful_kills": int, "rounds_participated": set()}
player_stats = defaultdict(lambda: {
    "total_kills": 0,
    "exit_frags": 0,
    "meaningful_kills": 0,
    "rounds_participated": set()  # Store (demo_name, round_num) tuples
})
countknife = 0

for demo_path in demo_files:
    print(f"Parsing {demo_path.name}")
    try:
        demo = Demo(str(demo_path), verbose=False)
        demo.parse(player_props=["name"])
        
        rounds_df = demo.rounds.to_pandas()
        kills_df = demo.kills.to_pandas()
        bombs_df = demo.bomb.to_pandas()
        damages_df = demo.damages.to_pandas()
        ticks_df = demo.ticks.to_pandas()
    except Exception as e:
        print(f"[warn] failed on {demo_path.name}: {e}")
        continue

    if rounds_df is None or rounds_df.empty:
        print(f"[warn] {demo_path.name} has empty rounds data")
        continue

    # Get tickrate
    tickrate = get_tickrate_from_header(demo, DEFAULT_TICKRATE)

    # ===== Remove knife/warmup round =====
    if not rounds_df.empty and damages_df is not None and not damages_df.empty:
        first_round = int(rounds_df["round_num"].min())
        first_round_damages = damages_df[damages_df["round_num"] == first_round]
        
        if not first_round_damages.empty and "weapon" in first_round_damages.columns:
            used_weapons = set(first_round_damages["weapon"].dropna().astype(str).str.lower().unique())
            
            if used_weapons.isdisjoint(valid_guns):
                print(f"[INFO] Removing knife round {first_round} from {demo_path.name}, weapons={used_weapons}")
                countknife += 1
                rounds_df = rounds_df[rounds_df["round_num"] != first_round]
                if kills_df is not None and not kills_df.empty:
                    kills_df = kills_df[kills_df["round_num"] != first_round]
                if bombs_df is not None and not bombs_df.empty:
                    bombs_df = bombs_df[bombs_df["round_num"] != first_round]
                if ticks_df is not None and not ticks_df.empty:
                    ticks_df = ticks_df[ticks_df["round_num"] != first_round]

    # ===== Track all players in all rounds from ticks =====
    if ticks_df is not None and not ticks_df.empty and "name" in ticks_df.columns and "round_num" in ticks_df.columns:
        ticks_df_clean = ticks_df.dropna(subset=["name", "round_num"]).copy()
        ticks_df_clean["name"] = ticks_df_clean["name"].apply(lambda x: norm_name(str(x)))
        
        # Get unique (player, round) combinations
        for _, tick_row in ticks_df_clean[["name", "round_num"]].drop_duplicates().iterrows():
            player = tick_row["name"]
            rnd = int(tick_row["round_num"])
            player_stats[player]["rounds_participated"].add((demo_path.name, rnd))

    # Check for required columns
    required_round_cols = ["round_num", "winner", "reason", "end"]
    
    missing_round = [c for c in required_round_cols if c not in rounds_df.columns]
    
    if missing_round:
        print(f"[warn] {demo_path.name} rounds missing columns: {missing_round}")
        continue
    
    # Process kills if available
    if kills_df is None or kills_df.empty:
        print(f"[info] {demo_path.name} has no kills data, only tracking round participation")
        continue
    
    required_kill_cols = ["attacker_name", "attacker_side", "tick", "round_num"]
    missing_kill = [c for c in required_kill_cols if c not in kills_df.columns]
    
    if missing_kill:
        print(f"[warn] {demo_path.name} kills missing columns: {missing_kill}")
        continue

    # ===== Build bomb event lookup: round_num -> {defuse_tick, detonate_tick} =====
    bomb_events = {}
    if bombs_df is not None and not bombs_df.empty and "event" in bombs_df.columns:
        for _, bomb_row in bombs_df.iterrows():
            rnd = bomb_row.get("round_num")
            event = str(bomb_row.get("event", "")).lower()
            tick = bomb_row.get("tick")
            
            if pd.isna(rnd) or pd.isna(tick):
                continue
            
            rnd = int(rnd)
            tick = int(tick)
            
            if rnd not in bomb_events:
                bomb_events[rnd] = {"defuse_tick": None, "detonate_tick": None}
            
            if event == "defuse":
                bomb_events[rnd]["defuse_tick"] = tick
            elif event == "detonate":
                bomb_events[rnd]["detonate_tick"] = tick

    # ===== Process each round =====
    for _, round_row in rounds_df.iterrows():
        round_num = round_row["round_num"]
        winner = str(round_row["winner"]).lower()
        reason = str(round_row["reason"]).lower()
        end_tick = round_row["end"]
        official_end_tick = round_row.get("official_end", end_tick)
        
        if pd.isna(round_num) or pd.isna(winner) or pd.isna(end_tick):
            continue
        
        round_num = int(round_num)
        end_tick = int(end_tick)
        if pd.notna(official_end_tick):
            official_end_tick = int(official_end_tick)
        else:
            official_end_tick = end_tick
        
        # Get all kills in this round
        round_kills = kills_df[kills_df["round_num"] == round_num].copy()
        
        if round_kills.empty:
            continue
        
        # Clean and normalize data
        round_kills = round_kills.dropna(subset=["attacker_name", "attacker_side"])
        round_kills["attacker_name"] = round_kills["attacker_name"].apply(lambda x: norm_name(str(x)))
        round_kills["attacker_side"] = round_kills["attacker_side"].apply(lambda x: str(x).lower())
        
        # Get bomb events for this round
        defuse_tick = None
        detonate_tick = None
        if round_num in bomb_events:
            defuse_tick = bomb_events[round_num]["defuse_tick"]
            detonate_tick = bomb_events[round_num]["detonate_tick"]
        
        # Process each kill
        for _, kill_row in round_kills.iterrows():
            attacker = kill_row["attacker_name"]
            attacker_side = kill_row["attacker_side"]
            kill_tick = int(kill_row["tick"])
            
            if attacker_side not in ["t", "ct"]:
                continue
            
            # Determine if this is an exit frag
            is_exit_frag = False
            
            # CT exit frags: CT lost by bomb exploding
            if attacker_side == "ct" and reason == "bomb_exploded" and winner == "t":
                # Use detonate tick if available, otherwise fallback to end_tick
                event_tick = detonate_tick if detonate_tick is not None else end_tick
                if kill_tick > event_tick:
                    is_exit_frag = True
            
            # T exit frags: T lost by bomb defused
            elif attacker_side == "t" and reason == "bomb_defused" and winner == "ct":
                if defuse_tick is not None and kill_tick > defuse_tick:
                    is_exit_frag = True
            
            # T exit frags: T lost by time running out
            elif attacker_side == "t" and reason == "time_ran_out" and winner == "ct":
                # Kill in the grace period (end < kill_tick <= official_end)
                if kill_tick > end_tick and kill_tick <= official_end_tick:
                    is_exit_frag = True
            
            # No exit frags possible for:
            # - ct_killed (T won by eliminating CT)
            # - t_killed (CT won by eliminating T)
            
            # Update statistics
            player_stats[attacker]["total_kills"] += 1
            
            if is_exit_frag:
                player_stats[attacker]["exit_frags"] += 1
            else:
                player_stats[attacker]["meaningful_kills"] += 1

# ===== Generate output =====
rows = []

for player, stats in player_stats.items():
    total_kills = stats["total_kills"]
    exit_frags = stats["exit_frags"]
    meaningful = stats["meaningful_kills"]
    total_rounds = len(stats["rounds_participated"])
    
    # Only include players who got at least 1 kill
    if total_kills == 0:
        continue
    
    exit_frag_rate = (exit_frags / total_kills) * 100
    meaningful_rate = (meaningful / total_kills) * 100
    
    rows.append({
        "Player": player,
        "TotalRounds": total_rounds,
        "TotalKills": total_kills,
        "MeaningfulKills": meaningful,
        "ExitFrags": exit_frags,
        "MeaningfulRate_%": round(meaningful_rate, 2),
        "ExitFragRate_%": round(exit_frag_rate, 2)
    })

if rows:
    df = pd.DataFrame(rows)
    # Sort by exit frag rate (descending) to highlight the "merchants"
    df = df.sort_values("ExitFragRate_%", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Results saved to {output_csv}")
    print(f"knife rounds removed: {countknife}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TOP 10 EXIT FRAG MERCHANTS:")
    print("="*70)
    top_exit = df.nlargest(10, "ExitFragRate_%")
    for idx, row in top_exit.iterrows():
        print(f"{row['Player']:.<25} {row['ExitFragRate_%']:>6.2f}% exit frags "
              f"({row['ExitFrags']}/{row['TotalKills']} kills, {row['TotalRounds']} rounds)")
    
    print("\n" + "="*70)
    print("TOP 10 MOST IMPACTFUL PLAYERS:")
    print("="*70)
    top_meaningful = df.nlargest(10, "MeaningfulRate_%")
    for idx, row in top_meaningful.iterrows():
        print(f"{row['Player']:.<25} {row['MeaningfulRate_%']:>6.2f}% meaningful "
              f"({row['MeaningfulKills']}/{row['TotalKills']} kills, {row['TotalRounds']} rounds)")
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS:")
    print("="*70)
    total_all_kills = df["TotalKills"].sum()
    total_exit = df["ExitFrags"].sum()
    total_meaningful = df["MeaningfulKills"].sum()
    avg_exit_rate = (total_exit / total_all_kills) * 100 if total_all_kills > 0 else 0
    
    print(f"Total kills analyzed: {total_all_kills}")
    print(f"Total exit frags: {total_exit} ({avg_exit_rate:.2f}%)")
    print(f"Total meaningful kills: {total_meaningful} ({100-avg_exit_rate:.2f}%)")
    print(f"Players analyzed: {len(df)}")
else:
    print("No data collected.")