import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline
import numpy as np

# === Configuration ===
input_csv = "weapon_economy_percentage.csv"
output_plot = "weapon_economy_timeline_smooth_falc.png"

# === MANUAL CONFIGURATION: Define which players to plot and their colors ===
players_to_plot = {
    # "torzsi": "#ffdcdc",      # Blue
    # "molodoy": "#f5cece",        # Orange
    "m0NESY": "#37b41d",
    # "910": "#faefd1",          # Green
    # "sh1ro": "#ddfde3",        # Purple
    # "w0nderful": "#fbfcd8",    # Brown
    # "broky": "#FEDCDC",         # Pink
    # "ZywOo": "#fafad8",        # Gray
    "NiKo": "#1f77b4",
    "kyousuke": "#ff7f0e",
    "TeSeS": "#d7d122",
    "kyxsan": "#bd4bc1",
    "Magisk": "#6ebfcf",
    # "molodoy": "#e377c2",
}

player_color_segments = {
    "m0NESY": {
        "early": {
            "color": "#E35555",  # Red for early events
            "events": [
                "BLAST_Bounty_2025_Season_1_Finals",
                "IEM_Katowice_2025",
                "PGL_Cluj-Napoca_2025",
                "ESL_Pro_League_Season_21",
                "BLAST_Open_Lisbon_2025",
                "PGL_Bucharest_2025"
            ]
        },
        "late": {
            "color": "#37b41d",  # Green for later events
        }
    }
}

# === MANUAL CONFIGURATION: Define event order for timeline ===
# List events in chronological order (left to right on plot)
event_order = [
    "BLAST_Bounty_2025_Season_1_Finals",
    "IEM_Katowice_2025",
    "PGL_Cluj-Napoca_2025",
    "ESL_Pro_League_Season_21",
    "BLAST_Open_Lisbon_2025",
    "PGL_Bucharest_2025",
    "IEM_Melbourne_2025",
    "BLAST_Rivals_2025_Season_1",
    "PGL_Astana_2025",
    "IEM_Dallas_2025",
    "BLAST.tv_Austin_Major_2025",
    "IEM_Cologne_2025",
    "BLAST_Bounty_2025_Season_2_Finals",
    "Esports_World_Cup_2025",
    "BLAST_Open_London_2025_Finals",
    "FISSURE_Playground_2",
    "ESL_Pro_League_Season_22",
    "IEM_Chengdu_2025",
    "BLAST_Rivals_2025_Season_2",
    "StarLadder_Budapest_Major_2025"
]

# === Load data ===
df = pd.read_csv(input_csv)

# Strip whitespace from Event column
df["Event"] = df["Event"].str.strip()

# Filter out "overall" rows for plotting
df_plot = df[df["Event"] != "overall"].copy()

magisk_excluded_events = ["FISSURE_Playground_2", "ESL_Pro_League_Season_22", "IEM_Chengdu_2025"]
df_plot = df_plot[~((df_plot["Player"] == "Magisk") & (df_plot["Event"].isin(magisk_excluded_events)))]

# Filter for only the players we want to plot
df_plot = df_plot[df_plot["Player"].isin(players_to_plot.keys())]

if df_plot.empty:
    print("No data found for the specified players!")
    exit()

# Get all unique events from data
all_events = df_plot["Event"].unique()

# Use provided event order, and append any events not in the order list
# Use ONLY the events specified in event_order
events_in_data = [e for e in event_order if e in all_events]
final_event_order = events_in_data

# Also filter the dataframe to only include these events
df_plot = df_plot[df_plot["Event"].isin(final_event_order)]

print(f"Events in timeline order: {final_event_order}")

# Create event to position mapping
event_to_position = {event: idx for idx, event in enumerate(final_event_order)}

# === Create plot ===
fig, ax = plt.subplots(figsize=(14, 8))
for player, color in players_to_plot.items():
    player_data = df_plot[df_plot["Player"] == player].copy()
    
    if player_data.empty:
        print(f"Warning: No data found for player {player}")
        continue
    
    # Map events to positions and sort
    player_data["event_position"] = player_data["Event"].map(event_to_position)
    player_data = player_data.sort_values("event_position")
    
    # Check if this player has color segments
    if player in player_color_segments:
        early_events = player_color_segments[player]["early"]["events"]
        early_color = player_color_segments[player]["early"]["color"]
        late_color = player_color_segments[player]["late"]["color"]
        
        # Split data into early and late
        early_data = player_data[player_data["Event"].isin(early_events)].copy()
        late_data = player_data[~player_data["Event"].isin(early_events)].copy()
        
        # Get full dataset for smooth connection
        x_all = player_data["event_position"].values
        y_all = player_data["AvgPercentageOfTeam"].values
        
        # Create full smooth line for proper connection
        if len(x_all) >= 3:
            x_smooth_all = np.linspace(x_all.min(), x_all.max(), 300)
            spl = make_interp_spline(x_all, y_all, k=3)
            y_smooth_all = spl(x_smooth_all)
            
            # Find transition point between early and late
            if not early_data.empty and not late_data.empty:
                early_max_pos = early_data["event_position"].max()
                late_min_pos = late_data["event_position"].min()
                
                # Plot early segment with label (includes transition to late_min_pos)
                early_mask = x_smooth_all <= late_min_pos
                ax.plot(x_smooth_all[early_mask], y_smooth_all[early_mask], 
                       label=f"G2 {player}", color=early_color, linewidth=2.5, alpha=0.8, zorder=3)
                x_early = early_data["event_position"].values
                y_early = early_data["AvgPercentageOfTeam"].values
                ax.plot(x_early, y_early, marker='o', color=early_color, markersize=6, 
                       linestyle='', markeredgewidth=1.5, markeredgecolor='white', zorder=4)
                
                # Plot late segment with label (starts FROM late_min_pos, no overlap)
                late_mask = x_smooth_all >= late_min_pos
                ax.plot(x_smooth_all[late_mask], y_smooth_all[late_mask], 
                       label=f"Falcons {player}", color=late_color, linewidth=2.5, alpha=0.8, zorder=2)
                x_late = late_data["event_position"].values
                y_late = late_data["AvgPercentageOfTeam"].values
                ax.plot(x_late, y_late, marker='o', color=late_color, markersize=6, 
                       linestyle='', markeredgewidth=1.5, markeredgecolor='white', zorder=4)
            elif not early_data.empty:
                # Only early data
                ax.plot(x_smooth_all, y_smooth_all, 
                       label=f"G2 {player}", color=early_color, linewidth=2.5, alpha=0.8)
                x_early = early_data["event_position"].values
                y_early = early_data["AvgPercentageOfTeam"].values
                ax.plot(x_early, y_early, marker='o', color=early_color, markersize=6, 
                       linestyle='', markeredgewidth=1.5, markeredgecolor='white')
            elif not late_data.empty:
                # Only late data
                ax.plot(x_smooth_all, y_smooth_all, 
                       label=f"Falcons {player}", color=late_color, linewidth=2.5, alpha=0.8)
                x_late = late_data["event_position"].values
                y_late = late_data["AvgPercentageOfTeam"].values
                ax.plot(x_late, y_late, marker='o', color=late_color, markersize=6, 
                       linestyle='', markeredgewidth=1.5, markeredgecolor='white')
    
    else:
        # Original plotting code for other players
        x = player_data["event_position"].values
        y = player_data["AvgPercentageOfTeam"].values
        
        if len(x) >= 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, label=player, color=color, linewidth=2.5, alpha=0.8)
            ax.plot(x, y, marker='o', color=color, markersize=6, linestyle='',
                   markeredgewidth=1.5, markeredgecolor='white')
        else:
            ax.plot(x, y, marker='o', label=player, color=color, linewidth=2, markersize=8)

# Formatting
ax.set_xlabel("Event", fontsize=12, fontweight='bold')
ax.set_ylabel("Avg % of Team Weapon Value", fontsize=12, fontweight='bold')
ax.set_title("Falcons - Player Weapon Economy Percentage Timeline", fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
# ax.set_ylim(23, 30)

# Set x-axis ticks to show event names at correct positions
ax.set_xticks(range(len(final_event_order)))
ax.set_xticklabels(final_event_order, rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {output_plot}")

plt.show()