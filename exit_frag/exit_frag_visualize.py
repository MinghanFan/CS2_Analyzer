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

# ===== TEAM COLORS =====
# Team name -> background color (hex)
TEAM_COLORS = {
    "Spirit": "#979da0",
    "Vitality": "#ead967",
    "Falcons": "#93ba9c",
    "G2": "#8c91a0",
    "FaZe": "#d16767",
    "Liquid": "#6789ae",
    "FURIA": "#877874",
    "MOUZ": "#d79696",
    "The MongolZ": "#d0ac7b",
    "Natus Vincere": "#e8d166",
    "GamerLegion": "#9bbddd",
    "Aurora": "#8fccd7",
    "Eternal Fire": "#d9bd81",
    "paiN": "#b7a4a8",
    "HEROIC": "#c6b5a0",
    "Astralis": "#d99a9a",
    "MIBR": "#e6cb66",
    "Passion UA": "#91a8d0",
    "Complexity": "#8a9ec2",
    "3DMAX": "#de7e7e",
    "Virtus.pro": "#e4a552",
    # Default fallback
    "default": "#73808a"
}

# ===== PLAYER TO TEAM MAPPING =====
# Based on HLTV data provided
PLAYER_TEAMS = {
    "donk": "Spirit",
    "ZywOo": "Vitality",
    "m0NESY": "Falcons",
    "sh1ro": "Spirit",
    "Twistzz": "FaZe",
    "KSCERATO": "FURIA",
    "ropz": "Vitality",
    "kyousuke": "Falcons",
    "frozen": "FaZe",
    "XANTARES": "Eternal Fire",
    "molodoy": "FURIA",
    "MATYS": "G2",
    "NiKo": "Falcons",
    "HeavyGod": "G2",
    "Grim": "Passion UA",
    "flameZ": "Vitality",
    "Spinx": "MOUZ",
    "Senzu": "The MongolZ",
    "b1t": "Natus Vincere",
    "EliGE": "Liqiud",
    "iM": "Natus Vincere",
    "YEKINDAR": "FURIA",
    "REZ": "GamerLegion",
    "Wicadia": "Eternal Fire",
    "yuurih": "FURIA",
    "zont1x": "Spirit",
    "PR": "GamerLegion",
    "dgt": "paiN",
    "xertioN": "MOUZ",
    "mezii": "Vitality",
    "tN1R": "HEROIC",
    "w0nderful": "Natus Vincere",
    "torzsi": "MOUZ",
    "jL": "Natus Vincere",
    "woxic": "Eternal Fire",
    "stavn": "Astralis",
    "insani": "MIBR",
    "hallzerk": "Complexity",
    "malbsMd": "G2",
    "Staehr": "Astralis",
    "nqz": "paiN",
    "mzinho": "The MongolZ",
    "Maka": "3DMAX",
    "Jimpphat": "MOUZ",
    "910": "The MongolZ",
    "fame": "Virtus.pro",
    "jottAAA": "Eternal Fire",
    "FL1T": "Virtus.pro",
    "NertZ": "Liquid",
    "rain": "FaZe",
    "magixx": "Spirit",
    "degster": "Falcons",
    "TeSeS": "Falcons",
    "SunPayus": "HEROIC",
    "device": "Astralis",
    "huNter-": "G2",
    "ICY": "Virtus.pro",
    "ultimate": "Liquid",
    "JT": "Complexity",
    "Ex3rcice": "3DMAX",
    "jabbi": "Astralis",
    "bLitz": "The MongolZ",
    "Tauson": "GamerLegion",
    "biguzera": "paiN",
    "broky": "FaZe",
    "Lucky": "3DMAX",
    "saffee": "MIBR",
    "dav1deuS": "paiN",
    "exit": "MIBR",
    "Techno": "The MongolZ",
    "bodyy": "3DMAX",
    "Magisk": "Falcons",
    "kyxsan": "Falcons",
    "Brollan": "MOUZ",
    "FL4MUS": "Virtus.pro",
    "yxngstxr": "HEROIC",
    "zweih": "Spirit",
    "apEX": "Vitality",
    "nicx": "Complexity",
    "snow": "paiN",
    "FalleN": "FURIA",
    "sl3nd": "GamerLegion",
    "Graviti": "3DMAX",
    "Aleksib": "Natus Vincere",
    "NAF": "Liquid",
    "electroNic": "Virtus.pro",
    "HooXi": "Astralis",
    "Snax": "G2",
    "chopper": "Spirit",
    "karrigan": "FaZe",
    "ztr": "GamerLegion",
    "LNZ": "HEROIC",
    "siuhy": "Liquid",
    "Lucaozy": "MIBR",
    "MAJ3R": "Eternal Fire",
    "cadiaN": "Astralis",
}

# ===== ABSOLUTE POSITION CONTROLS - CHANGE THESE VALUES =====
# Figure dimensions (pixels at 300 DPI)
FIGURE_WIDTH_INCHES = 11
FIGURE_HEIGHT_INCHES = 10

# X-axis absolute positions (0-100 scale, like percentage of width)
PHOTO_X = 8              # Photo left edge position
NAME_X = 14.5              # Player name position
BAR_START_X = 32-3         # Where bars start
BAR_END_X = 75-3           # Where bars end (max length) - SHORTENED (was 80)
STATS_X_OFFSET = 2       # Stats offset from bar end

# Y-axis spacing
Y_START = 12             # First player from top (%)
Y_SPACING = 9            # Space between each player (%)

# Element sizes
PHOTO_SIZE = 57          # Photo size in pixels
BAR_HEIGHT = 1.7         # Bar height (relative) - INCREASED
PHOTO_BORDER = 1       # Photo border width
BAR_BORDER = 2           # Bar border width

# Font sizes
NAME_FONT = 12*1.3
STATS_FONT = 10
TITLE_FONT = 20*1.7
SUBTITLE_FONT = 9

# Title positions (0-100 scale)
TITLE_X = 53
TITLE_Y = 98             # MOVED HIGHER (was 96)
SUBTITLE_Y = 95          # MOVED HIGHER (was 93)

# Colors (soft but a touch brighter)
RED_BAR = "#e17c7c"
RED_BORDER = "#c47d7d"
GREEN_BAR = "#8bd0a7"
GREEN_BORDER = '#8ab7a0'

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

def load_player_image(player_name, photo_dir, size=50, team_color='#2c3e50'):
    photo_path = photo_dir / f"{player_name}.png"
    img = None
    if photo_path.exists():
        try:
            img = Image.open(photo_path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
        except Exception:
            img = None
    if img is None:
        img = create_placeholder_image(size)
    
    # Scale PNG only; leave background color to AnnotationBbox
    max_dim = max(img.width, img.height)
    if max_dim:
        zoom = min(size / max_dim, 1.2)
    else:
        zoom = 1.0
    
    return OffsetImage(img, zoom=zoom)

def create_exit_frag_chart(data, title, subtitle, output_file, is_top=True):
    """Create chart with ABSOLUTE positioning - no automatic adjustments"""
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    
    # Remove all axes
    ax.set_xlim(0, 105)
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
        
        # Add grey background for odd rows (idx 1, 3, 5, 7, 9)
        if idx % 2 == 0:
            grey_rect = plt.Rectangle((0, y_pos - 4.475), 107, BAR_HEIGHT*5.29,
                                     facecolor="#f2f2f2", edgecolor='none',
                                     zorder=0, transform=ax.transData, clip_on=False)
            ax.add_patch(grey_rect)
        
        # 1. Add photo at FIXED X position with team color background
        team_name = PLAYER_TEAMS.get(player_name, "default")
        team_color = TEAM_COLORS.get(team_name, TEAM_COLORS["default"])
        
        img = load_player_image(player_name, player_photos_dir, size=PHOTO_SIZE, team_color=team_color)
        imagebox = AnnotationBbox(img, (PHOTO_X, y_pos), 
                                 frameon=True, 
                                 box_alignment=(0.5, 0.5),
                                 bboxprops=dict(edgecolor=bar_edge, 
                                               linewidth=PHOTO_BORDER, 
                                               facecolor=team_color))
        ax.add_artist(imagebox)
        
        # 2. Add player name at FIXED X position
        ax.text(NAME_X, y_pos, player_name, 
               va='center', ha='left', 
               fontsize=NAME_FONT, fontweight='bold',
               color='#4a545b', transform=ax.transData)
        
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
               fontsize=STATS_FONT, color='#4b5964', transform=ax.transData)
    
    # 5. Add title at FIXED position
    ax.text(TITLE_X, TITLE_Y, title,
           ha='center', va='center',
           fontsize=TITLE_FONT, fontweight='bold', color='#4a545b',
           transform=ax.transData)
    
    # 6. Add subtitle at FIXED position
    ax.text(TITLE_X, SUBTITLE_Y, subtitle,
           ha='center', va='center',
           fontsize=SUBTITLE_FONT, style='italic', color='#7f8c8d',
           transform=ax.transData)
    
    # 7. Add minimum rounds note
    ax.text(101, 2, f'Minimum {MIN_ROUNDS} rounds played',
           ha='right', va='bottom',
           fontsize=8, style='italic', color='#9aa3aa',
           transform=ax.transData)
    
    # 8. Add credits
    ax.text(101, 0.5, 'Photo by HLTV | Data by clu0ki',
           ha='right', va='bottom',
           fontsize=7, color='#9aa3aa',
           transform=ax.transData)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight', facecolor='white')
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
top_title = "TOP 10 EXIT-FRAGGERS"
top_subtitle = ("Exit frags = kills AFTER team already lost the round (bomb exploded/defused, time ran out)\n"
                "These players get the most meaningless kills when rounds are already decided")
create_exit_frag_chart(top10, top_title, top_subtitle, output_top10, is_top=True)

bottom_title = "TOP 10 NON EXIT-FRAGGERS"
bottom_subtitle = ("Exit frags = kills AFTER team already lost the round (bomb exploded/defused, time ran out)\n"
                   "These players have the lowest rate of meaningless kills - every kill counts")
create_exit_frag_chart(bottom10, bottom_title, bottom_subtitle, output_bottom10, is_top=False)

print("\nVisualizations complete!")
print(f"Top 10 merchants: {output_top10}")
print(f"Bottom 10 cleanest: {output_bottom10}")
