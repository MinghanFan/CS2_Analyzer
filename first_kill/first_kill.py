from awpy import Demo
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# ===== Configuration =====
demo_root = Path("/Volumes/TOSHIBA EXT/Demo_2025")  # Change to your root directory
output_csv = "first_kill/first_kill_analysis.csv"
DEFAULT_TICKRATE = 64

# ===== Valid weapons for knife round detection =====
valid_guns = {
    "hkp2000", "elite", "glock", "p250", "fiveseven", "tec9", "cz75a", "deagle", "revolver", "usp_silencer",
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

# ===== Team mapping (you can expand this) =====
PLAYER_TEAMS = {
    "donk": "Spirit",
    "ZywOo": "Vitality",
    "m0NESY": "Falcons",
    "sh1ro": "Spirit",
    "Twistzz": "FaZe",
    "KSCERATO": "FURIA",
    "ropz": "Vitality",
    "frozen": "FaZe",
    "molodoy": "FURIA",
    "NiKo": "Falcons",
    "HeavyGod": "G2",
    "flameZ": "Vitality",
    "Spinx": "MOUZ",
    "b1t": "Natus Vincere",
    "iM": "Natus Vincere",
    "YEKINDAR": "FURIA",
    "yuurih": "FURIA",
    "zont1x": "Spirit",
    "mezii": "Vitality",
    "w0nderful": "Natus Vincere",
    "torzsi": "MOUZ",
    "jL": "Natus Vincere",
    "malbsMd": "G2",
    "Jimpphat": "MOUZ",
    "910": "The MongolZ",
    "mzinho": "The MongolZ",
    "device": "Astralis",
    "huNter-": "G2",
    "broky": "FaZe",
    "Brollan": "MOUZ",
    "apEX": "Vitality",
    "Aleksib": "Natus Vincere",
    "karrigan": "FaZe",
    "Snax": "G2",
    "chopper": "Spirit",
    "magixx": "Spirit",
    "zweih": "Spirit",
    "tN1r": "Spirit",
    "Magisk": "Falcons",
    "TeSeS": "Falcons",
    "Staehr": "Astralis",
    "jabbi": "Astralis",
    "HooXi": "Astralis",
    "kyxsan": "Falcons",
    "xertioN": "MOUZ",
    "bLitz": "The MongolZ",
    "Techno": "The MongolZ",
    "MATYS": "G2",
    "kyousuke": "Falcons"
}

# ===== Global aggregators =====
# player -> {stats dict}
player_stats = defaultdict(lambda: {
    "team": "Unknown",
    "rounds_played": 0,
    "total_kills": 0,
    "first_kills": 0,
    "rounds_won": 0,
    "rounds_lost": 0,
    "fk_and_won": 0,      # Got first kill AND won round
    "fk_and_lost": 0,     # Got first kill BUT lost round
    "no_fk_and_won": 0,   # No first kill BUT won round
    "no_fk_and_lost": 0,  # No first kill AND lost round
})

countknife = 0

# ===== Find demo files =====
demo_files = [f for f in sorted(demo_root.rglob("*.dem")) if not f.name.startswith("._")]
print(f"Found {len(demo_files)} demos under {demo_root}")

for demo_path in demo_files:
    print(f"Parsing {demo_path.name}")
    try:
        demo = Demo(str(demo_path), verbose=False)
        demo.parse(player_props=["name", "team_name"])
        
        rounds_df = demo.rounds.to_pandas()
        kills_df = demo.kills.to_pandas()
        damages_df = demo.damages.to_pandas()
        ticks_df = demo.ticks.to_pandas()
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
                if kills_df is not None and not kills_df.empty:
                    kills_df = kills_df[kills_df["round_num"] != first_round]
                if ticks_df is not None and not ticks_df.empty:
                    ticks_df = ticks_df[ticks_df["round_num"] != first_round]

    # ===== Track team assignments from ticks =====
    player_to_team = {}
    if ticks_df is not None and not ticks_df.empty:
        if "name" in ticks_df.columns and "team_name" in ticks_df.columns:
            ticks_clean = ticks_df.dropna(subset=["name", "team_name"]).copy()
            ticks_clean["name"] = ticks_clean["name"].apply(lambda x: norm_name(str(x)))
            
            # Get the most common team_name for each player
            for player_name in ticks_clean["name"].unique():
                player_ticks = ticks_clean[ticks_clean["name"] == player_name]
                most_common_team = player_ticks["team_name"].mode()
                if len(most_common_team) > 0:
                    player_to_team[player_name] = str(most_common_team.iloc[0])

    # Check for required columns
    required_round_cols = ["round_num", "winner"]
    missing_round = [c for c in required_round_cols if c not in rounds_df.columns]
    
    if missing_round:
        print(f"[warn] {demo_path.name} rounds missing columns: {missing_round}")
        continue
    
    # Process kills if available
    if kills_df is None or kills_df.empty:
        print(f"[info] {demo_path.name} has no kills data")
        continue
    
    required_kill_cols = ["attacker_name", "attacker_side", "tick", "round_num"]
    missing_kill = [c for c in required_kill_cols if c not in kills_df.columns]
    
    if missing_kill:
        print(f"[warn] {demo_path.name} kills missing columns: {missing_kill}")
        continue

    # ===== Process each round =====
    for _, round_row in rounds_df.iterrows():
        round_num = round_row["round_num"]
        winner = str(round_row["winner"]).lower()
        
        if pd.isna(round_num) or pd.isna(winner):
            continue
        
        round_num = int(round_num)
        
        # Get all kills in this round
        round_kills = kills_df[kills_df["round_num"] == round_num].copy()
        
        if round_kills.empty:
            continue
        
        # Clean and normalize data
        round_kills = round_kills.dropna(subset=["attacker_name", "attacker_side"])
        round_kills["attacker_name"] = round_kills["attacker_name"].apply(lambda x: norm_name(str(x)))
        round_kills["attacker_side"] = round_kills["attacker_side"].apply(lambda x: str(x).lower())
        round_kills = round_kills.sort_values("tick")
        
        # Get first kill of the round
        first_kill = round_kills.iloc[0] if len(round_kills) > 0 else None
        first_killer = None
        first_killer_side = None
        
        if first_kill is not None:
            first_killer = first_kill["attacker_name"]
            first_killer_side = first_kill["attacker_side"]
        
        # Get all unique players in this round from ticks
        round_players = set()
        if ticks_df is not None and not ticks_df.empty:
            round_ticks = ticks_df[ticks_df["round_num"] == round_num]
            if not round_ticks.empty and "name" in round_ticks.columns and "side" in round_ticks.columns:
                ticks_players = round_ticks.dropna(subset=["name", "side"])
                for _, tick_row in ticks_players.iterrows():
                    player_name = norm_name(str(tick_row["name"]))
                    player_side = str(tick_row["side"]).lower()
                    if player_side in ["t", "ct"]:
                        round_players.add((player_name, player_side))
        
        # Update stats for all players in this round
        for player_name, player_side in round_players:
            # Determine team name
            team_name = "Unknown"
            if player_name in PLAYER_TEAMS:
                team_name = PLAYER_TEAMS[player_name]
            elif player_name in player_to_team:
                team_name = player_to_team[player_name]
            
            player_stats[player_name]["team"] = team_name
            player_stats[player_name]["rounds_played"] += 1
            
            # Check if player won this round
            player_won = (player_side == winner)
            
            if player_won:
                player_stats[player_name]["rounds_won"] += 1
            else:
                player_stats[player_name]["rounds_lost"] += 1
            
            # Check if player got first kill
            got_first_kill = (player_name == first_killer)
            
            if got_first_kill:
                player_stats[player_name]["first_kills"] += 1
                
                if player_won:
                    player_stats[player_name]["fk_and_won"] += 1
                else:
                    player_stats[player_name]["fk_and_lost"] += 1
            else:
                if player_won:
                    player_stats[player_name]["no_fk_and_won"] += 1
                else:
                    player_stats[player_name]["no_fk_and_lost"] += 1
        
        # Count total kills for each player in this round
        for _, kill_row in round_kills.iterrows():
            attacker = kill_row["attacker_name"]
            player_stats[attacker]["total_kills"] += 1

# ===== Generate output =====
rows = []

for player, stats in player_stats.items():
    rounds_played = stats["rounds_played"]
    
    # Only include players who played at least 1 round
    if rounds_played == 0:
        continue
    
    total_kills = stats["total_kills"]
    first_kills = stats["first_kills"]
    rounds_won = stats["rounds_won"]
    rounds_lost = stats["rounds_lost"]
    fk_and_won = stats["fk_and_won"]
    fk_and_lost = stats["fk_and_lost"]
    no_fk_and_won = stats["no_fk_and_won"]
    no_fk_and_lost = stats["no_fk_and_lost"]
    team = stats["team"]
    
    # Calculate rates
    first_kill_rate = (first_kills / rounds_played) * 100 if rounds_played > 0 else 0
    win_rate = (rounds_won / rounds_played) * 100 if rounds_played > 0 else 0
    
    # First kill impact
    fk_win_rate = (fk_and_won / first_kills) * 100 if first_kills > 0 else 0
    no_fk_win_rate = (no_fk_and_won / (rounds_played - first_kills)) * 100 if (rounds_played - first_kills) > 0 else 0
    
    rows.append({
        "Player": player,
        "Team": team,
        "RoundsPlayed": rounds_played,
        "TotalKills": total_kills,
        "FirstKills": first_kills,
        "RoundsWon": rounds_won,
        "RoundsLost": rounds_lost,
        "FK_and_Won": fk_and_won,
        "FK_and_Lost": fk_and_lost,
        "NoFK_and_Won": no_fk_and_won,
        "NoFK_and_Lost": no_fk_and_lost,
        "FirstKillRate_%": round(first_kill_rate, 2),
        "WinRate_%": round(win_rate, 2),
        "FK_WinRate_%": round(fk_win_rate, 2),
        "NoFK_WinRate_%": round(no_fk_win_rate, 2),
    })

if rows:
    df = pd.DataFrame(rows)
    # Sort by first kills (descending)
    df = df.sort_values("FirstKills", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Results saved to {output_csv}")
    print(f"Knife rounds removed: {countknife}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TOP 10 FIRST KILL LEADERS:")
    print("="*80)
    top_fk = df.nlargest(10, "FirstKills")
    for idx, row in top_fk.iterrows():
        print(f"{row['Player']:.<20} {row['Team']:.<15} "
              f"{row['FirstKills']:>3} first kills in {row['RoundsPlayed']:>4} rounds "
              f"({row['FirstKillRate_%']:>5.2f}%) | FK Win Rate: {row['FK_WinRate_%']:>5.2f}%")
    
    print("\n" + "="*80)
    print("TOP 10 FIRST KILL RATE (min 0 rounds):")
    print("="*80)
    df_qualified = df[df["RoundsPlayed"] >= 0]
    if len(df_qualified) >= 10:
        top_rate = df_qualified.nlargest(10, "FirstKillRate_%")
        for idx, row in top_rate.iterrows():
            print(f"{row['Player']:.<20} {row['Team']:.<15} "
                  f"{row['FirstKillRate_%']:>5.2f}% ({row['FirstKills']}/{row['RoundsPlayed']} rounds) | "
                  f"FK Win Rate: {row['FK_WinRate_%']:>5.2f}%")
    
    print("\n" + "="*80)
    print("LOW 10 FIRST KILL RATE (min 0 rounds):")
    print("="*80)
    df_qualified = df[df["RoundsPlayed"] >= 0]
    if len(df_qualified) >= 10:
        top_rate = df_qualified.nsmallest(10, "FirstKillRate_%")
        for idx, row in top_rate.iterrows():
            print(f"{row['Player']:.<20} {row['Team']:.<15} "
                  f"{row['FirstKillRate_%']:>5.2f}% ({row['FirstKills']}/{row['RoundsPlayed']} rounds) | "
                  f"FK Win Rate: {row['FK_WinRate_%']:>5.2f}%")
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS:")
    print("="*80)
    total_rounds = df["RoundsPlayed"].sum()
    total_first_kills = df["FirstKills"].sum()
    total_fk_wins = df["FK_and_Won"].sum()
    total_fk_losses = df["FK_and_Lost"].sum()
    total_no_fk_wins = df["NoFK_and_Won"].sum()
    total_no_fk_losses = df["NoFK_and_Lost"].sum()
    
    overall_fk_win_rate = (total_fk_wins / total_first_kills) * 100 if total_first_kills > 0 else 0
    rounds_without_fk = total_rounds - total_first_kills
    overall_no_fk_win_rate = (total_no_fk_wins / rounds_without_fk) * 100 if rounds_without_fk > 0 else 0
    
    print(f"Total rounds analyzed: {total_rounds}")
    print(f"Total first kills: {total_first_kills}")
    print(f"Win rate WITH first kill: {overall_fk_win_rate:.2f}%")
    print(f"Win rate WITHOUT first kill: {overall_no_fk_win_rate:.2f}%")
    print(f"First kill advantage: +{overall_fk_win_rate - overall_no_fk_win_rate:.2f}%")
    print(f"Players analyzed: {len(df)}")
else:
    print("No data collected.")