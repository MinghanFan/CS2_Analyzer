import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline
import numpy as np

# === Configuration ===
input_csv = "weapon_economy_percentage.csv"
output_plot = "weapon_economy_timeline_straight.png"

# === MANUAL CONFIGURATION: Define which players to plot and their colors ===
players_to_plot = {
    "torzsi": "#ffdcdc",
    "molodoy": "#cfcef5",
    "m0NESY": "#37b41d",
    "910": "#faefd0",
    "sh1ro": "#d9fddf",
    "w0nderful": "#fbffc7",
    #"broky": "#FFD3D3",
    "ZywOo": "#f2fcbf",
}

player_color_segments = {
    # "m0NESY": {
    #     "early": {
    #         "color": "#E35555",  # Red for early events
    #         "events": [
    #             "BLAST_Bounty_2025_Season_1_Finals",
    #             "IEM_Katowice_2025",
    #             "PGL_Cluj-Napoca_2025",
    #             "ESL_Pro_League_Season_21",
    #             "BLAST_Open_Lisbon_2025",
    #             "PGL_Bucharest_2025",
    #         ]
    #     },
    #     "late": {
    #         "color": "#37b41d",  # Green for later events
    #     }
    # }
}

# === MANUAL CONFIGURATION: Define event order for timeline ===
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
df["Event"] = df["Event"].str.strip()
df_plot = df[df["Event"] != "overall"].copy()

magisk_excluded_events = ["FISSURE_Playground_2", "ESL_Pro_League_Season_22", "IEM_Chengdu_2025"]
df_plot = df_plot[~((df_plot["Player"] == "Magisk") & (df_plot["Event"].isin(magisk_excluded_events)))]

df_plot = df_plot[df_plot["Player"].isin(players_to_plot.keys())]

if df_plot.empty:
    print("No data found for the specified players!")
    exit()

all_events = df_plot["Event"].unique()
events_in_data = [e for e in event_order if e in all_events]
final_event_order = events_in_data
df_plot = df_plot[df_plot["Event"].isin(final_event_order)]

print(f"Events in timeline order: {final_event_order}")

event_to_position = {event: idx for idx, event in enumerate(final_event_order)}

# === Create plot ===
fig, ax = plt.subplots(figsize=(14, 8))

for player, color in players_to_plot.items():
    player_data = df_plot[df_plot["Player"] == player].copy()
    
    if player_data.empty:
        print(f"Warning: No data found for player {player}")
        continue
    
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
        
        # Plot with straight lines (no spline interpolation)
        if not early_data.empty and not late_data.empty:
            # Plot early segment with straight lines
            x_early = early_data["event_position"].values
            y_early = early_data["AvgPercentageOfTeam"].values
            ax.plot(x_early, y_early, label=f"G2 {player}", color=early_color, 
                   linewidth=2.5, alpha=0.8, zorder=3, marker='o', markersize=6,
                   markeredgewidth=1.5, markeredgecolor='white')
            
            # Plot late segment with straight lines
            x_late = late_data["event_position"].values
            y_late = late_data["AvgPercentageOfTeam"].values
            ax.plot(x_late, y_late, label=f"Falcons {player}", color=late_color, 
                   linewidth=2.5, alpha=0.8, zorder=2, marker='o', markersize=6,
                   markeredgewidth=1.5, markeredgecolor='white')
            
        elif not early_data.empty:
            x_early = early_data["event_position"].values
            y_early = early_data["AvgPercentageOfTeam"].values
            ax.plot(x_early, y_early, label=f"G2 {player}", color=early_color, 
                   linewidth=2.5, alpha=0.8, marker='o', markersize=6,
                   markeredgewidth=1.5, markeredgecolor='white')
            
        elif not late_data.empty:
            x_late = late_data["event_position"].values
            y_late = late_data["AvgPercentageOfTeam"].values
            ax.plot(x_late, y_late, label=f"Falcons {player}", color=late_color, 
                   linewidth=2.5, alpha=0.8, marker='o', markersize=6,
                   markeredgewidth=1.5, markeredgecolor='white')
    
    else:
        # Straight lines for all other players too
        x = player_data["event_position"].values
        y = player_data["AvgPercentageOfTeam"].values
        
        ax.plot(x, y, label=player, color=color, linewidth=2.5, alpha=0.8,
               marker='o', markersize=6, markeredgewidth=1.5, markeredgecolor='white')

# Formatting
ax.set_xlabel("Event", fontsize=12, fontweight='bold')
ax.set_ylabel("Avg % of Team Weapon Value", fontsize=12, fontweight='bold')
ax.set_title("Awper Weapon Economy Percentage Timeline", fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

ax.set_xticks(range(len(final_event_order)))
ax.set_xticklabels(final_event_order, rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {output_plot}")

plt.show()