from awpy import Demo
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from awpy.plot.utils import game_to_pixel

# ===== Configuration =====
demo_root = Path("/Volumes/TOSHIBA EXT/last3month")  # Change to your root directory
output_dir = Path("first_blood_heatmaps")
output_dir.mkdir(exist_ok=True)

# ===== Valid weapons for knife round detection =====
valid_guns = {
    "hkp2000", "elite", "glock", "p250", "fiveseven", "tec9", "cz75a", "deagle", "revolver",
    "mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon",
    "galilar", "famas", "ak47", "m4a1", "m4a1_silencer", "sg556", "aug",
    "ssg08", "awp", "g3sg1", "scar20",
    "nova", "xm1014", "mag7", "sawedoff",
    "negev", "m249", "taser"
}

# ===== Find demo files =====
demo_files = [f for f in sorted(demo_root.rglob("*.dem")) if not f.name.startswith("._")]
print(f"Found {len(demo_files)} demos under {demo_root}")

# ===== Global aggregators =====
# map_name -> list of (X, Y, Z, side) tuples for first kills
first_blood_positions = defaultdict(list)
countknife = 0

for demo_path in demo_files:
    print(f"Parsing {demo_path.name}")
    try:
        demo = Demo(str(demo_path), verbose=False)
        demo.parse()
        
        rounds_df = demo.rounds.to_pandas()
        kills_df = demo.kills.to_pandas()
        damages_df = demo.damages.to_pandas()
        header = demo.header
    except Exception as e:
        print(f"[warn] failed on {demo_path.name}: {e}")
        continue

    if rounds_df is None or rounds_df.empty:
        print(f"[warn] {demo_path.name} has empty rounds data")
        continue

    # Get map name
    map_name = header.get("map_name", "unknown")
    if map_name == "unknown":
        print(f"[warn] Could not determine map name for {demo_path.name}")
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

    # Check for required columns
    if kills_df is None or kills_df.empty:
        print(f"[info] {demo_path.name} has no kills data")
        continue
    
    required_kill_cols = ["attacker_X", "attacker_Y", "attacker_Z", "attacker_side", "tick", "round_num"]
    missing_kill = [c for c in required_kill_cols if c not in kills_df.columns]
    
    if missing_kill:
        print(f"[warn] {demo_path.name} kills missing columns: {missing_kill}")
        continue

    # ===== Process each round to find first blood =====
    for round_num in rounds_df["round_num"].unique():
        round_kills = kills_df[kills_df["round_num"] == round_num].copy()
        
        if round_kills.empty:
            continue
        
        # Sort by tick to get first kill
        round_kills = round_kills.sort_values("tick")
        
        # Get the first kill of the round
        first_kill = round_kills.iloc[0]
        
        # Extract position and side
        x = first_kill["attacker_X"]
        y = first_kill["attacker_Y"]
        z = first_kill["attacker_Z"]
        side = str(first_kill["attacker_side"]).lower()
        
        # Skip if any coordinate is NaN
        if pd.isna(x) or pd.isna(y) or pd.isna(z) or pd.isna(side):
            continue
        
        # Store the position
        first_blood_positions[map_name].append((x, y, z, side))

# ===== Generate heatmaps for each map =====
print(f"\n{'='*70}")
print("GENERATING HEATMAPS")
print(f"{'='*70}")
print(f"Knife rounds removed: {countknife}")

# Try to import awpy map data
try:
    import awpy.data.map_data
    MAP_DATA = awpy.data.map_data.MAP_DATA
except:
    print("[ERROR] Could not load map data from awpy")
    MAP_DATA = {}

for map_name, positions in first_blood_positions.items():
    if not positions:
        continue
    
    print(f"\nGenerating heatmap for {map_name} ({len(positions)} first bloods)")
    
    # Separate by side
    ct_positions = [(x, y, z) for x, y, z, side in positions if side == "ct"]
    t_positions = [(x, y, z) for x, y, z, side in positions if side == "t"]
    all_positions = [(x, y, z) for x, y, z, _, in positions]
    
    # Check if map is in awpy data
    if map_name not in MAP_DATA:
        print(f"[warn] Map {map_name} not found in awpy map data, skipping visualization")
        continue
    
    # Create visualizations using scatter plots
    try:
        from awpy.plot import plot
        from awpy.plot.utils import game_to_pixel
        
        # Overall visualization
        if len(all_positions) > 0:
            fig, ax = plot(map_name=map_name)
            
            # Convert game coordinates to pixel coordinates
            for x, y, z in all_positions:
                px = game_to_pixel(map_name, (x, y, z))[0]
                py = game_to_pixel(map_name, (x, y, z))[1]
                
                # Plot as red circle with transparency
                ax.plot(px, py, 'o', color='orange', markersize=3, alpha=0.3)
            
            plt.title(f"First Blood Locations - {map_name.upper()} (All Sides)", 
                     fontsize=16, color='white', pad=20)
            output_file = output_dir / f"{map_name}_firstblood_all.png"
            plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='black')
            plt.close()
            print(f"  Saved: {output_file}")
        
        # CT-side visualization
        if len(ct_positions) > 0:
            fig, ax = plot(map_name=map_name)
            
            # Convert game coordinates to pixel coordinates
            for x, y, z in ct_positions:
                px = game_to_pixel(map_name, (x, y, z))[0]
                py = game_to_pixel(map_name, (x, y, z))[1]
                
                # Plot as blue circle with transparency
                ax.plot(px, py, 'o', color='blue', markersize=3, alpha=0.3, markeredgewidth=0)
            
            plt.title(f"First Blood Locations - {map_name.upper()} (CT Side)", 
                     fontsize=16, color='white', pad=20)
            output_file = output_dir / f"{map_name}_firstblood_ct.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            print(f"  Saved: {output_file}")
        
        # T-side visualization
        if len(t_positions) > 0:
            fig, ax = plot(map_name=map_name)
            
            # Convert game coordinates to pixel coordinates
            for x, y, z in t_positions:
                px = game_to_pixel(map_name, (x, y, z))[0]
                py = game_to_pixel(map_name, (x, y, z))[1]
                
                # Plot as orange circle with transparency
                ax.plot(px, py, 'o', color='orange', markersize=3, alpha=0.3, markeredgewidth=0)
            
            plt.title(f"First Blood Locations - {map_name.upper()} (T Side)", 
                     fontsize=16, color='white', pad=20)
            output_file = output_dir / f"{map_name}_firstblood_t.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            print(f"  Saved: {output_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to create heatmap for {map_name}: {e}")
        continue

# ===== Generate summary statistics =====
print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")

summary_rows = []
for map_name, positions in first_blood_positions.items():
    ct_count = len([p for p in positions if p[3] == "ct"])
    t_count = len([p for p in positions if p[3] == "t"])
    total = len(positions)
    
    ct_pct = (ct_count / total * 100) if total > 0 else 0
    t_pct = (t_count / total * 100) if total > 0 else 0
    
    summary_rows.append({
        "Map": map_name,
        "Total_FirstBloods": total,
        "CT_FirstBloods": ct_count,
        "T_FirstBloods": t_count,
        "CT_Percentage": round(ct_pct, 2),
        "T_Percentage": round(t_pct, 2)
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Total_FirstBloods", ascending=False)
    
    # Save summary
    summary_csv = output_dir / "first_blood_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
    
    # Print summary
    print("\nFirst Blood Statistics by Map:")
    print(summary_df.to_string(index=False))
    
    # Overall statistics
    total_first_bloods = summary_df["Total_FirstBloods"].sum()
    total_ct = summary_df["CT_FirstBloods"].sum()
    total_t = summary_df["T_FirstBloods"].sum()
    
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total first bloods analyzed: {total_first_bloods}")
    print(f"CT first bloods: {total_ct} ({total_ct/total_first_bloods*100:.2f}%)")
    print(f"T first bloods: {total_t} ({total_t/total_first_bloods*100:.2f}%)")
    print(f"Maps analyzed: {len(summary_df)}")
    print(f"\nHeatmaps saved to: {output_dir.absolute()}")
else:
    print("No data collected.")