import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

# ===== Configuration =====
csv_file = "exit_frag/exit_frag_analysis.csv"
player_photos_dir = Path("player_photos")
output_top10 = "exit_frag_merchants_top10.png"
output_bottom10 = "exit_frag_cleanest_bottom10.png"
MIN_ROUNDS = 1000

# ===== ABSOLUTE POSITION CONTROLS - CHANGE THESE VALUES =====
# Figure dimensions (pixels at 300 DPI)
FIGURE_WIDTH_INCHES = 11
FIGURE_HEIGHT_INCHES = 10

# X-axis absolute positions (0-100 scale, like percentage of width)
PHOTO_X = 8              # Photo left edge position
NAME_X = 18              # Player name position
BAR_START_X = 32         # Where bars start
BAR_END_X = 80           # Where bars end (max length)
STATS_X_OFFSET = 2       # Stats offset from bar end

# Y-axis spacing
Y_START = 12              # First player from top (%)
Y_SPACING = 9            # Space between each player (%)

# Element sizes
PHOTO_SIZE = 55          # Photo size in pixels
BAR_HEIGHT = 1.7         # Bar height (relative) - INCREASED
PHOTO_BORDER = 1.5       # Photo border width
BAR_BORDER = 2           # Bar border width

# Font sizes
NAME_FONT = 12
STATS_FONT = 10
TITLE_FONT = 20
SUBTITLE_FONT = 9

# Title positions (0-100 scale)
TITLE_X = 55
TITLE_Y = 98             # MOVED HIGHER (was 96)
SUBTITLE_Y = 95          # MOVED HIGHER (was 93)

# Colors
RED_BAR = '#e74c3c'
RED_BORDER = '#c0392b'
GREEN_BAR = '#2ecc71'
GREEN_BORDER = '#27ae60'

# ===== Manual Adjustments =====
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
    return df

def create_placeholder_image(size=100):
    img = Image.new('RGB', (size, size), color='#34495e')
    draw = ImageDraw.Draw(img)
    draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill='#95a5a6')
    return img

def load_player_image(player_name, photo_dir, size=50):
    photo_path = photo_dir / f"{player_name}.png"
    img = None
    if photo_path.exists():
        try:
            img = Image.open(photo_path)
            if img.mode == 'RGBA':
                img = img.convert('RGBA')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception:
            img = None
    if img is None:
        img = create_placeholder_image(size)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return OffsetImage(img, zoom=1.0)

def create_exit_frag_chart(data, title, subtitle, output_file, is_top=True):
    """Create chart with ABSOLUTE positioning - no automatic adjustments"""
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    
    # Remove all axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Colors
    bar_color = RED_BAR if is_top else GREEN_BAR
    bar_edge = RED_BORDER if is_top else GREEN_BORDER
    
    # Keep original order (highest/lowest already at top from sorting)
    # Do NOT reverse
    
    # Calculate exit rates
    exit_rates = [(row["ExitFrags"] / row["TotalKills"]) * 100 for _, row in data.iterrows()]
    max_rate = max(exit_rates)
    
    # Draw each player row
    for idx, (i, row) in enumerate(data.iterrows()):
        player_name = row["Player"]
        exit_frags = int(row["ExitFrags"])
        total_kills = int(row["TotalKills"])
        total_rounds = int(row["TotalRounds"])
        exit_rate = (exit_frags / total_kills) * 100
        
        # Calculate Y position (fixed spacing)
        y_pos = 100 - Y_START - (idx * Y_SPACING)
        
        # 1. Add photo at FIXED X position
        img = load_player_image(player_name, player_photos_dir, size=PHOTO_SIZE)
        imagebox = AnnotationBbox(img, (PHOTO_X, y_pos), 
                                 frameon=True, 
                                 box_alignment=(0.5, 0.5),
                                 bboxprops=dict(edgecolor=bar_edge, 
                                               linewidth=PHOTO_BORDER, 
                                               facecolor='white'))
        ax.add_artist(imagebox)
        
        # 2. Add player name at FIXED X position
        ax.text(NAME_X, y_pos, player_name, 
               va='center', ha='left', 
               fontsize=NAME_FONT, fontweight='bold',
               color='#2c3e50', transform=ax.transData)
        
        # 3. Add bar - FIXED start, length based on rate (normalized to max)
        bar_length = (exit_rate / max_rate) * (BAR_END_X - BAR_START_X)
        bar_rect = plt.Rectangle((BAR_START_X, y_pos - BAR_HEIGHT/2), bar_length, BAR_HEIGHT,
                                 facecolor=bar_color, edgecolor=bar_edge, 
                                 linewidth=BAR_BORDER, transform=ax.transData)
        ax.add_patch(bar_rect)
        
        # 4. Add stats at FIXED offset from bar end
        stats_x = BAR_START_X + bar_length + STATS_X_OFFSET
        stats_text = f"{exit_rate:.3f}%  |  {exit_frags}/{total_kills} kills  |  {total_rounds} rounds"
        ax.text(stats_x, y_pos, stats_text,
               va='center', ha='left',
               fontsize=STATS_FONT, color='#34495e', transform=ax.transData)
    
    # 5. Add title at FIXED position
    ax.text(TITLE_X, TITLE_Y, title,
           ha='center', va='center',
           fontsize=TITLE_FONT, fontweight='bold', color='#2c3e50',
           transform=ax.transData)
    
    # 6. Add subtitle at FIXED position
    ax.text(TITLE_X, SUBTITLE_Y, subtitle,
           ha='center', va='center',
           fontsize=SUBTITLE_FONT, style='italic', color='#7f8c8d',
           transform=ax.transData)
    
    # 7. Add minimum rounds note
    ax.text(110, 2, f'Minimum {MIN_ROUNDS} rounds played',
           ha='right', va='bottom',
           fontsize=8, style='italic', color='#95a5a6',
           transform=ax.transData)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()

# ===== Main execution =====
df = pd.read_csv(csv_file)
df = apply_manual_adjustments(df, MANUAL_ADJUSTMENTS)
df["PreciseExitRate"] = (df["ExitFrags"] / df["TotalKills"]) * 100
df_filtered = df[df["TotalRounds"] >= MIN_ROUNDS].copy()

print(f"Total players: {len(df)}")
print(f"Players with >={MIN_ROUNDS} rounds: {len(df_filtered)}")

if len(df_filtered) < 10:
    print(f"[ERROR] Not enough players")
    exit(1)

top10 = df_filtered.nlargest(10, "PreciseExitRate").reset_index(drop=True)
bottom10 = df_filtered.nsmallest(10, "PreciseExitRate").reset_index(drop=True)

# Generate visualizations
top_title = "TOP 10 EXIT FRAGGERS"
top_subtitle = ("Exit frags = kills AFTER team already lost the round (bomb exploded/defused, time ran out)\n"
                "These players get the most meaningless kills when rounds are already decided")
create_exit_frag_chart(top10, top_title, top_subtitle, output_top10, is_top=True)

bottom_title = "TOP 10 NON-EXIT FRAGGERS"
bottom_subtitle = ("Exit frags = kills AFTER team already lost the round (bomb exploded/defused, time ran out)\n"
                   "These players have the lowest rate of meaningless kills - every kill counts")
create_exit_frag_chart(bottom10, bottom_title, bottom_subtitle, output_bottom10, is_top=False)

print("\nVisualizations complete!")
print(f"Top 10 merchants: {output_top10}")
print(f"Bottom 10 cleanest: {output_bottom10}")