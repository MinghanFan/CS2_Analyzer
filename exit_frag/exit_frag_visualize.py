import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# ===== Configuration =====
csv_file = "exit_frag/exit_frag_analysis.csv"  # Your exit frag CSV
player_photos_dir = Path("player_photos")  # Directory containing player photos (name.png)
output_top10 = "exit_frag_merchants_top10.png"
output_bottom10 = "exit_frag_cleanest_bottom10.png"
MIN_ROUNDS = 1000  # Minimum rounds to be included

# ===== Manual Adjustments =====
# Format: "PlayerName": {"ExitFrags": +1, "TotalKills": -1, "MeaningfulKills": 0, "TotalRounds": +5}
# Use this to manually adjust numbers if needed (+/- values)
MANUAL_ADJUSTMENTS = {
    "ICY": {"ExitFrags": 0, "TotalKills": -16, "MeaningfulKills": 0, "TotalRounds": +108},
    "FL1T": {"ExitFrags": 0, "TotalKills": -20, "MeaningfulKills": 0, "TotalRounds": +108},
    "FL4MUS": {"ExitFrags": 0, "TotalKills": -15, "MeaningfulKills": 0, "TotalRounds": +44},
    "sl3nd": {"ExitFrags": 0, "TotalKills": -13, "MeaningfulKills": 0, "TotalRounds": +16},
    "SunPayus": {"ExitFrags": 0, "TotalKills": -7, "MeaningfulKills": 0, "TotalRounds": +120},
    "KSCERATO": {"ExitFrags": 0, "TotalKills": -15, "MeaningfulKills": 0, "TotalRounds": +319},
    "ropz": {"ExitFrags": 0, "TotalKills": -20, "MeaningfulKills": 0, "TotalRounds": +582},
    "nqz": {"ExitFrags": 0, "TotalKills": -14, "MeaningfulKills": 0, "TotalRounds": +85},
    "Aleksib": {"ExitFrags": 0, "TotalKills": -15, "MeaningfulKills": 0, "TotalRounds": +57},
    "Twistzz": {"ExitFrags": 0, "TotalKills": -17, "MeaningfulKills": 0, "TotalRounds": +76},
    "b1t": {"ExitFrags": 0, "TotalKills": -18, "MeaningfulKills": 0, "TotalRounds": +57},
    "zweih": {"ExitFrags": 0, "TotalKills": -11, "MeaningfulKills": 0, "TotalRounds": +81},
    "chopper": {"ExitFrags": 0, "TotalKills": -28, "MeaningfulKills": 0, "TotalRounds": +237},
    "EliGE": {"ExitFrags": 0, "TotalKills": -14, "MeaningfulKills": 0, "TotalRounds": +72},
    "dav1deuS": {"ExitFrags": 0, "TotalKills": -14, "MeaningfulKills": 0, "TotalRounds": +85},
    "bodyy": {"ExitFrags": 0, "TotalKills": -14, "MeaningfulKills": 0, "TotalRounds": +166},
    "snow": {"ExitFrags": 0, "TotalKills": -14, "MeaningfulKills": 0, "TotalRounds": +85},
    "TeSeS": {"ExitFrags": 0, "TotalKills": -26, "MeaningfulKills": 0, "TotalRounds": +488},
    "jottAAA": {"ExitFrags": 0, "TotalKills": -24, "MeaningfulKills": 0, "TotalRounds": +201},
    "kyousuke": {"ExitFrags": 0, "TotalKills": -9, "MeaningfulKills": 0, "TotalRounds": +83},
}

def apply_manual_adjustments(df, adjustments):
    """Apply manual adjustments to the dataframe"""
    df = df.copy()
    for player, adj in adjustments.items():
        if player in df["Player"].values:
            idx = df[df["Player"] == player].index[0]
            if "ExitFrags" in adj:
                df.loc[idx, "ExitFrags"] += adj["ExitFrags"]
            if "TotalKills" in adj:
                df.loc[idx, "TotalKills"] += adj["TotalKills"]
            if "MeaningfulKills" in adj:
                df.loc[idx, "MeaningfulKills"] += adj["MeaningfulKills"]
            if "TotalRounds" in adj:
                df.loc[idx, "TotalRounds"] += adj["TotalRounds"]
            print(f"[ADJUSTED] {player}: ExitFrags={df.loc[idx, 'ExitFrags']}, "
                  f"TotalKills={df.loc[idx, 'TotalKills']}, MeaningfulKills={df.loc[idx, 'MeaningfulKills']}, "
                  f"TotalRounds={df.loc[idx, 'TotalRounds']}")
    return df

# ===== Load data =====
df = pd.read_csv(csv_file)

# Apply manual adjustments
df = apply_manual_adjustments(df, MANUAL_ADJUSTMENTS)

# Calculate precise exit rate for sorting
df["PreciseExitRate"] = (df["ExitFrags"] / df["TotalKills"]) * 100

# Filter players with enough rounds
df_filtered = df[df["TotalRounds"] >= MIN_ROUNDS].copy()

print(f"Total players: {len(df)}")
print(f"Players with >={MIN_ROUNDS} rounds: {len(df_filtered)}")

if len(df_filtered) < 10:
    print(f"[ERROR] Not enough players with >={MIN_ROUNDS} rounds. Only {len(df_filtered)} players found.")
    exit(1)

# Sort and get top/bottom 10 based on PRECISE exit rate
top10 = df_filtered.nlargest(10, "PreciseExitRate").reset_index(drop=True)
bottom10 = df_filtered.nsmallest(10, "PreciseExitRate").reset_index(drop=True)

print("\nTop 10 Exit Frag Merchants:")
print(top10[["Player", "PreciseExitRate", "ExitFrags", "TotalKills", "TotalRounds"]])

print("\nBottom 10 (Cleanest Players):")
print(bottom10[["Player", "PreciseExitRate", "ExitFrags", "TotalKills", "TotalRounds"]])

# ===== Helper function to create placeholder image =====
def create_placeholder_image(size=100):
    """Create a simple placeholder image"""
    img = Image.new('RGB', (size, size), color='#34495e')
    draw = ImageDraw.Draw(img)
    # Draw a simple silhouette or question mark
    draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill='#95a5a6')
    return img

# ===== Helper function to load player photo =====
def load_player_image(player_name, photo_dir, size=50):
    """Load player photo, return OffsetImage or placeholder if not found"""
    photo_path = photo_dir / f"{player_name}.png"
    
    img = None
    if photo_path.exists():
        try:
            img = Image.open(photo_path)
            # Convert to RGBA if PNG with transparency, otherwise RGB
            if img.mode == 'RGBA':
                img = img.convert('RGBA')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load image for {player_name}: {e}")
            img = None
    
    if img is None:
        print(f"[WARN] Photo not found for {player_name}, using placeholder")
        img = create_placeholder_image(size)
    
    # Resize to square with high quality (LANCZOS provides best quality)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return OffsetImage(img, zoom=1.0)

# ===== Create visualization function =====
def create_exit_frag_chart(data, title, subtitle, output_file, is_top=True):
    """
    Create horizontal bar chart with player photos
    
    Args:
        data: DataFrame with player stats
        title: Main title
        subtitle: Subtitle explaining methodology
        output_file: Output filename
        is_top: True for top merchants (red), False for cleanest (green)
    """
    fig, ax = plt.subplots(figsize=(10, 10))  # Reduced width from 12 to 10
    
    # Colors
    if is_top:
        bar_color = '#e74c3c'  # Red for merchants
        bar_edge = '#c0392b'
    else:
        bar_color = '#2ecc71'  # Green for clean players
        bar_edge = '#27ae60'
    
    # Reverse order so highest/lowest is on top
    data = data.iloc[::-1].reset_index(drop=True)
    
    y_positions = np.arange(len(data))
    # Use precise exit rate for bar lengths
    exit_rates = [(row["ExitFrags"] / row["TotalKills"]) * 100 for _, row in data.iterrows()]
    
    # Create horizontal bars
    bars = ax.barh(y_positions, exit_rates, height=0.7, 
                   color=bar_color, edgecolor=bar_edge, linewidth=2)
    
    # Add player photos on the left
    for idx, (i, row) in enumerate(data.iterrows()):
        player_name = row["Player"]
        
        # Load and add player photo (with placeholder fallback)
        img = load_player_image(player_name, player_photos_dir, size=50)
        # Position photo closer (reduced from -1.4 to -1.1)
        imagebox = AnnotationBbox(img, (-1.1, idx), 
                                 frameon=True, 
                                 box_alignment=(0.5, 0.5),
                                 bboxprops=dict(edgecolor=bar_edge, 
                                               linewidth=1.5, 
                                               facecolor='white'))
        ax.add_artist(imagebox)
        
        # Add player name closer to photo (reduced from -0.7 to -0.5)
        ax.text(-0.5, idx, player_name, 
               va='center', ha='left', 
               fontsize=11, fontweight='bold',
               color='#2c3e50')
        
        # Add detailed stats on the bar (using potentially adjusted values)
        exit_frags = int(row["ExitFrags"])
        total_kills = int(row["TotalKills"])
        total_rounds = int(row["TotalRounds"])
        exit_rate = (exit_frags / total_kills) * 100
        
        # Stats text closer to the bar (reduced from 0.3 to 0.15)
        stats_text = f"{exit_rate:.3f}%  |  {exit_frags}/{total_kills} kills  |  {total_rounds} rounds"
        ax.text(exit_rate + 0.15, idx, stats_text,
               va='center', ha='left',
               fontsize=10, color='#34495e', fontweight='normal')
    
    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])  # Remove y-axis labels (we have player names)
    ax.set_xlabel('Exit Frag Rate (%)', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_xlim(-1.3, max(exit_rates) * 1.25)  # Reduced left margin from -1.7 to -1.3
    
    # Title and subtitle - moved to the right (x position from 0.5 to 0.65)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#2c3e50', x=0.65)
    ax.text(0.65, 1.08, subtitle, 
           transform=ax.transAxes,
           fontsize=10, ha='center', style='italic', color='#7f8c8d',
           wrap=True)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add a note about minimum rounds at bottom
    fig.text(0.99, 0.01, f'Minimum {MIN_ROUNDS} rounds played', 
            ha='right', va='bottom', fontsize=8, style='italic', color='#95a5a6')
    
    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()

# ===== Generate visualizations =====

# Top 10 Exit Frag Merchants
top_title = "TOP 10 EXIT FRAGGERS"
top_subtitle = ("Exit frags = kills AFTER team already lost the round (bomb exploded/defused, time ran out)\n"
                "These players get the most meaningless kills when rounds are already decided")
create_exit_frag_chart(top10, top_title, top_subtitle, output_top10, is_top=True)

# Bottom 10 Cleanest Players
bottom_title = "TOP 10 LEAST EXIT FRAGGERS"
bottom_subtitle = ("Exit frags = kills AFTER team already lost the round (bomb exploded/defused, time ran out)\n"
                   "These players have the lowest rate of meaningless kills - every kill counts")
create_exit_frag_chart(bottom10, bottom_title, bottom_subtitle, output_bottom10, is_top=False)

print("\n✅ Visualizations complete!")
print(f"📊 Top 10 merchants: {output_top10}")
print(f"📊 Bottom 10 cleanest: {output_bottom10}")