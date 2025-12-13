import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === Configuration ===
input_csv = "weapon_economy_percentage.csv"
output_plot = "weapon_economy_timeline.png"

# === MANUAL CONFIGURATION: Define which players to plot and their colors ===
players_to_plot = {
    "torzsi": "#1f77b4",      # Blue
    "molodoy": "#ff7f0e",        # Orange
    "m0NESY": "#d62728",
    "910": "#2ca02c",          # Green
    "sh1ro": "#9467bd",        # Purple
    "w0nderful": "#8c564b",    # Brown
    # "broky": "#e377c2",         # Pink
    # "ZywOo": "#7f7f7f",        # Gray
}

# === MANUAL CONFIGURATION: Define event order for timeline ===
# List events in chronological order (left to right on plot)
event_order = [
    "BLAST_Bounty_2025_Season_1_Finals",
    "IEM_Katowice_2025",
    "PGL_Cluj-Napoca_2025",
    "ESL_Pro_League_Season_21_2025",
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
    "BLAST_Open_London _2025_Finals",
    "FISSURE_Playground_2",
    "ESL_Pro_League_Season_22",
    "IEM_Chengdu_2025",
    "BLAST_Rivals_2025_Season_2",
]

# === Load data ===
df = pd.read_csv(input_csv)

# Strip whitespace from Event column
df["Event"] = df["Event"].str.strip()

# Filter out "overall" rows for plotting
df_plot = df[df["Event"] != "overall"].copy()

# Filter for only the players we want to plot
df_plot = df_plot[df_plot["Player"].isin(players_to_plot.keys())]

if df_plot.empty:
    print("No data found for the specified players!")
    exit()

# Get all unique events from data
all_events = df_plot["Event"].unique()

# Use provided event order, and append any events not in the order list
events_in_data = [e for e in event_order if e in all_events]
events_not_in_order = [e for e in all_events if e not in event_order]
final_event_order = events_in_data + sorted(events_not_in_order)

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
    
    # Plot using numeric positions
    ax.plot(
        player_data["event_position"],
        player_data["AvgPercentageOfTeam"],
        marker='o',
        label=player,
        color=color,
        linewidth=2,
        markersize=8
    )

# Formatting
ax.set_xlabel("Event", fontsize=12, fontweight='bold')
ax.set_ylabel("Avg % of Team Weapon Value", fontsize=12, fontweight='bold')
ax.set_title("Player Weapon Economy Percentage Timeline", fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_ylim(23, 30) 

# Set x-axis ticks to show event names at correct positions
ax.set_xticks(range(len(final_event_order)))
ax.set_xticklabels(final_event_order, rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {output_plot}")

plt.show()