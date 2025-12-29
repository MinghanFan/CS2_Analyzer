from awpy import Demo
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# ===== Configuration =====
demo_root = Path("/Volumes/TOSHIBA EXT/Demo_2025")  # Change to your root directory
#demo_root = Path("/Users/minghanfan/Documents/Test/train") 
output_csv = "weapon_advantage_analysis.csv"
ECONOMY_THRESHOLD = 2000  # Configurable threshold for "equal" economy
DEFAULT_TICKRATE = 64

# ===== Valid weapons for knife round detection =====
valid_guns = {
    "hkp2000", "elite", "glock", "p250", "fiveseven", "tec9", "cz75a", "deagle", "revolver", "usp_silencer", "usp_silencer_off",
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

# ===== Global aggregators =====
# player -> condition -> {"kills": int, "deaths": int, "rounds": int}
# conditions: "advantage", "equal", "disadvantage", "overall"
player_stats = defaultdict(lambda: {
    "advantage": {"kills": 0, "deaths": 0, "rounds": 0},
    "equal": {"kills": 0, "deaths": 0, "rounds": 0},
    "disadvantage": {"kills": 0, "deaths": 0, "rounds": 0},
    "overall": {"kills": 0, "deaths": 0, "rounds": 0}
})

countknife = 0

for demo_path in demo_files:
    print(f"\nParsing {demo_path.name}")
    try:
        demo = Demo(str(demo_path), verbose=False)
        demo.parse(player_props=["name", "current_equip_value", "side"])
        
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

    # ===== Process each round =====
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
        ].dropna(subset=["name", "current_equip_value", "side"])
        
        if tick_slice.empty:
            continue
        
        # Get most common equipment value for each player
        tick_slice = tick_slice.copy()
        grouped = tick_slice.groupby(["name", "side"])["current_equip_value"].agg(
            lambda x: x.value_counts().idxmax() if not x.empty else 0
        ).reset_index()
        # print(f" Round {round_num}: Player equipment values:\n{grouped}")
        
        # Normalize names
        grouped["name"] = grouped["name"].apply(norm_name)
        
        # Calculate team totals for each side
        team_totals = grouped.groupby("side")["current_equip_value"].sum().to_dict()
        
        ct_total = team_totals.get("ct", 0)
        t_total = team_totals.get("t", 0)
        
        # Determine economy condition for each side
        # CT perspective
        if ct_total > t_total + ECONOMY_THRESHOLD:
            ct_condition = "advantage"
            t_condition = "disadvantage"
        elif ct_total < t_total - ECONOMY_THRESHOLD:
            ct_condition = "disadvantage"
            t_condition = "advantage"
        else:
            ct_condition = "equal"
            t_condition = "equal"
        
        # Create player -> condition mapping for this round
        player_conditions = {}
        for _, row in grouped.iterrows():
            player = row["name"]
            side = row["side"]
            condition = ct_condition if side == "ct" else t_condition
            player_conditions[player] = condition
            
            # Track round participation
            player_stats[player][condition]["rounds"] += 1
            player_stats[player]["overall"]["rounds"] += 1
        
        # Get kills for this round
        round_kills = kills_df[kills_df["round_num"] == round_num].copy()
        
        if round_kills.empty:
            continue
        
        # Process kills
        round_kills = round_kills.dropna(subset=["attacker_name", "victim_name"])
        round_kills["attacker_name"] = round_kills["attacker_name"].apply(norm_name)
        round_kills["victim_name"] = round_kills["victim_name"].apply(norm_name)
        
        for _, kill_row in round_kills.iterrows():
            attacker = kill_row["attacker_name"]
            victim = kill_row["victim_name"]
            
            # Track kills for attacker
            if attacker in player_conditions:
                condition = player_conditions[attacker]
                player_stats[attacker][condition]["kills"] += 1
                player_stats[attacker]["overall"]["kills"] += 1
            
            # Track deaths for victim
            if victim in player_conditions:
                condition = player_conditions[victim]
                player_stats[victim][condition]["deaths"] += 1
                player_stats[victim]["overall"]["deaths"] += 1

print(f"\n{'='*70}")
print("GENERATING OUTPUT CSV")
print(f"{'='*70}")

# ===== Generate output CSV =====
results = []

for player, conditions in player_stats.items():
    adv = conditions["advantage"]
    eq = conditions["equal"]
    disadv = conditions["disadvantage"]
    overall = conditions["overall"]
    
    # Only include players who participated in at least one round
    if overall["rounds"] == 0:
        continue
    
    results.append({
        "Player": player,
        
        # Advantage
        "Adv_Kills": adv["kills"],
        "Adv_Deaths": adv["deaths"],
        "Adv_Rounds": adv["rounds"],
        
        # Equal
        "Equal_Kills": eq["kills"],
        "Equal_Deaths": eq["deaths"],
        "Equal_Rounds": eq["rounds"],
        
        # Disadvantage
        "Disadv_Kills": disadv["kills"],
        "Disadv_Deaths": disadv["deaths"],
        "Disadv_Rounds": disadv["rounds"],
        
        # Overall
        "Overall_Kills": overall["kills"],
        "Overall_Deaths": overall["deaths"],
        "Overall_Rounds": overall["rounds"]
    })

# Create DataFrame and save
df = pd.DataFrame(results)
df = df.sort_values("Overall_Rounds", ascending=False)
df.to_csv(output_csv, index=False)

print(f"\nDone! Results saved to {output_csv}")
print(f"Knife rounds removed: {countknife}")
print(f"Total players tracked: {len(df)}")

# ===== Print summary statistics =====
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

if not df.empty:
    total_rounds_all = df["Overall_Rounds"].sum()
    total_adv_rounds = df["Adv_Rounds"].sum()
    total_eq_rounds = df["Equal_Rounds"].sum()
    total_disadv_rounds = df["Disadv_Rounds"].sum()
    
    print(f"Total player-rounds: {total_rounds_all}")
    print(f"  Advantage rounds: {total_adv_rounds} ({total_adv_rounds/total_rounds_all*100:.1f}%)")
    print(f"  Equal rounds: {total_eq_rounds} ({total_eq_rounds/total_rounds_all*100:.1f}%)")
    print(f"  Disadvantage rounds: {total_disadv_rounds} ({total_disadv_rounds/total_rounds_all*100:.1f}%)")
    
    # Calculate K/D ratios by condition (for players with significant data)
    significant_players = df[df["Overall_Rounds"] >= 20].copy()
    
    if not significant_players.empty:
        print("\n" + "="*70)
        print(f"K/D RATIOS (Players with 20+ rounds, n={len(significant_players)})")
        print("="*70)
        
        # Calculate K/D ratios
        for condition, prefix in [("Advantage", "Adv"), ("Equal", "Equal"), ("Disadvantage", "Disadv"), ("Overall", "Overall")]:
            kills_col = f"{prefix}_Kills"
            deaths_col = f"{prefix}_Deaths"
            rounds_col = f"{prefix}_Rounds"
            
            # Filter players with at least 5 rounds in this condition (or 20 for overall)
            min_rounds = 200 if condition == "Overall" else 50
            condition_df = significant_players[significant_players[rounds_col] >= min_rounds].copy()
            
            if condition_df.empty:
                continue
            
            condition_df["KD"] = condition_df[kills_col] / condition_df[deaths_col].replace(0, 1)
            avg_kd = condition_df["KD"].mean()
            
            print(f"{condition:.<20} Avg K/D: {avg_kd:.3f}")
        
        print("\n" + "="*70)
        print("TOP 10 PLAYERS BY OVERALL K/D (20+ rounds)")
        print("="*70)
        
        top_players = significant_players.copy()
        top_players["Overall_KD"] = top_players["Overall_Kills"] / top_players["Overall_Deaths"].replace(0, 1)
        top_players = top_players.nlargest(10, "Overall_KD")
        
        for _, row in top_players.iterrows():
            print(f"{row['Player']:.<25} {row['Overall_KD']:>5.2f} K/D "
                  f"({row['Overall_Kills']}/{row['Overall_Deaths']} in {row['Overall_Rounds']} rounds)")