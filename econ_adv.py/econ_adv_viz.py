import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

# ===== Configuration =====
csv_file = "weapon_advantage_analysis.csv"
player_photos_dir = Path("player_photos")
output_advantage = "weapon_adv_best_advantage.png"
output_equal = "weapon_adv_best_equal.png"
output_disadvantage = "weapon_adv_best_disadvantage.png"
output_consistent = "weapon_adv_most_consistent.png"
MIN_ROUNDS = 1200
MIN_CONDITION_ROUNDS = 400  # Minimum rounds in the specific condition being analyzed

# ===== TEAM COLORS =====
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
    "default": "#73808a"
}

# ===== PLAYER TO TEAM MAPPING =====
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
    "EliGE": "Liquid",
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

# ===== ABSOLUTE POSITION CONTROLS =====
FIGURE_WIDTH_INCHES = 11
FIGURE_HEIGHT_INCHES = 10

PHOTO_X = 8
NAME_X = 14.5
BAR_START_X = 29
BAR_END_X = 72
STATS_X_OFFSET = 2

Y_START = 12
Y_SPACING = 9

PHOTO_SIZE = 57
BAR_HEIGHT = 1.7
PHOTO_BORDER = 1
BAR_BORDER = 2

NAME_FONT = 12*1.3
STATS_FONT = 10
TITLE_FONT = 20*1.7
SUBTITLE_FONT = 9

TITLE_X = 53
TITLE_Y = 98
SUBTITLE_Y = 95

# ===== COLORS BY CONDITION =====
ADV_BAR = "#dda19e"
ADV_BORDER = "#c58b7d"

EQUAL_BAR = "#7bb3e0"
EQUAL_BORDER = "#6a9dca"

DISADV_BAR = "#8bd0a7"
DISADV_BORDER = "#8ab7a0"
CONSISTENT_BAR = "#b8a4d4"
CONSISTENT_BORDER = "#a594bf"

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
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
        except Exception:
            img = None
    if img is None:
        img = create_placeholder_image(size)
    
    max_dim = max(img.width, img.height)
    if max_dim:
        zoom = min(size / max_dim, 1.2)
    else:
        zoom = 1.0
    
    return OffsetImage(img, zoom=zoom)

def create_weapon_adv_chart(data, title, subtitle, output_file, metric_col, bar_color, border_color):
    """Create chart showing K/D performance in specific economy condition"""
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Calculate max K/D for bar scaling
    kd_values = [row[metric_col] for _, row in data.iterrows()]
    max_kd = max(kd_values)
    
    # Draw each player row
    for idx, (i, row) in enumerate(data.iterrows()):
        player_name = row["Player"]
        
        # Calculate K/Ds for all conditions
        adv_kd = row["Adv_KD"]
        eq_kd = row["Equal_KD"]
        disadv_kd = row["Disadv_KD"]
        
        # Get the primary metric for this chart
        primary_kd = row[metric_col]
        
        # Calculate Y position
        y_pos = 100 - Y_START - (idx * Y_SPACING)
        
        # Add grey background for even rows
        if idx % 2 == 0:
            grey_rect = plt.Rectangle((0, y_pos - 4.475), 107, BAR_HEIGHT*5.29,
                                     facecolor="#f2f2f2", edgecolor='none',
                                     zorder=0, transform=ax.transData, clip_on=False)
            ax.add_patch(grey_rect)
        
        # Add photo with team color background
        team_name = PLAYER_TEAMS.get(player_name, "default")
        team_color = TEAM_COLORS.get(team_name, TEAM_COLORS["default"])
        
        img = load_player_image(player_name, player_photos_dir, size=PHOTO_SIZE)
        imagebox = AnnotationBbox(img, (PHOTO_X, y_pos), 
                                 frameon=True, 
                                 box_alignment=(0.5, 0.5),
                                 bboxprops=dict(edgecolor=border_color, 
                                               linewidth=PHOTO_BORDER, 
                                               facecolor=team_color))
        ax.add_artist(imagebox)
        
        # Add player name
        ax.text(NAME_X, y_pos, player_name, 
               va='center', ha='left', 
               fontsize=NAME_FONT, fontweight='bold',
               color='#4a545b', transform=ax.transData)
        
        # Add bar (normalized to max K/D)
        bar_length = (primary_kd / max_kd) * (BAR_END_X - BAR_START_X)
        bar_rect = plt.Rectangle((BAR_START_X, y_pos - BAR_HEIGHT/2), bar_length, BAR_HEIGHT,
                                 facecolor=bar_color, edgecolor=border_color, 
                                 linewidth=BAR_BORDER, transform=ax.transData)
        ax.add_patch(bar_rect)
        
        # Add stats showing all three conditions
        stats_x = BAR_START_X + bar_length + STATS_X_OFFSET
        stats_text = f"Adv: {adv_kd:.2f}  |  Equal: {eq_kd:.2f}  |  Disadv: {disadv_kd:.2f}"
        ax.text(stats_x, y_pos, stats_text,
               va='center', ha='left',
               fontsize=STATS_FONT, color='#4b5964', transform=ax.transData)
    
    # Add title
    ax.text(TITLE_X, TITLE_Y, title,
           ha='center', va='center',
           fontsize=TITLE_FONT, fontweight='bold', color='#4a545b',
           transform=ax.transData)
    
    # Add subtitle
    ax.text(TITLE_X, SUBTITLE_Y, subtitle,
           ha='center', va='center',
           fontsize=SUBTITLE_FONT, style='italic', color='#7f8c8d',
           transform=ax.transData)
    
    # Add minimum rounds note
    ax.text(101.2, 2, f'Minimum {MIN_ROUNDS} total rounds, {MIN_CONDITION_ROUNDS} in condition',
           ha='right', va='bottom',
           fontsize=8, style='italic', color='#9aa3aa',
           transform=ax.transData)
    
    # Add credits
    ax.text(101.2, 0.5, 'Photo by HLTV | Data by clu0ki',
           ha='right', va='bottom',
           fontsize=7, color='#9aa3aa',
           transform=ax.transData)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, format='png', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()

# ===== Main execution =====
print("Loading data...")
df = pd.read_csv(csv_file)

# Calculate K/D ratios
df["Adv_KD"] = df["Adv_Kills"] / df["Adv_Deaths"].replace(0, 1)
df["Equal_KD"] = df["Equal_Kills"] / df["Equal_Deaths"].replace(0, 1)
df["Disadv_KD"] = df["Disadv_Kills"] / df["Disadv_Deaths"].replace(0, 1)
df["Overall_KD"] = df["Overall_Kills"] / df["Overall_Deaths"].replace(0, 1)

# Calculate K/D variance (for consistency metric)
df["KD_Variance"] = df[["Adv_KD", "Equal_KD", "Disadv_KD"]].var(axis=1)

print(f"Total players: {len(df)}")

# Filter: minimum total rounds
df_filtered = df[df["Overall_Rounds"] >= MIN_ROUNDS].copy()
print(f"Players with >={MIN_ROUNDS} total rounds: {len(df_filtered)}")

if len(df_filtered) < 10:
    print(f"[ERROR] Not enough players with {MIN_ROUNDS} rounds")
    exit(1)

# ===== Chart 1: Best in Advantage (BLUE) =====
print("\nGenerating Chart 1: Best in Advantage...")
df_adv = df_filtered[df_filtered["Adv_Rounds"] >= MIN_CONDITION_ROUNDS].copy()
print(f"  Players with >={MIN_CONDITION_ROUNDS} advantage rounds: {len(df_adv)}")

if len(df_adv) >= 10:
    top_adv = df_adv.nlargest(10, "Adv_KD").reset_index(drop=True)
    create_weapon_adv_chart(
        top_adv,
        "TOP 10 K/D - ADVANTAGE $",
        "Players with highest K/D when their team has economy advantage (±$2000 threshold)",
        output_advantage,
        "Adv_KD",
        ADV_BAR,
        ADV_BORDER
    )
else:
    print(f"  [SKIP] Not enough players for advantage chart")

# ===== Chart 2: Best in Equal (YELLOW) =====
print("\nGenerating Chart 2: Best in Equal...")
df_eq = df_filtered[df_filtered["Equal_Rounds"] >= MIN_CONDITION_ROUNDS].copy()
print(f"  Players with >={MIN_CONDITION_ROUNDS} equal rounds: {len(df_eq)}")

if len(df_eq) >= 10:
    top_eq = df_eq.nlargest(10, "Equal_KD").reset_index(drop=True)
    create_weapon_adv_chart(
        top_eq,
        "TOP 10 K/D - EQUAL $",
        "Players with highest K/D when economies are balanced (±$2000 threshold)",
        output_equal,
        "Equal_KD",
        EQUAL_BAR,
        EQUAL_BORDER
    )
else:
    print(f"  [SKIP] Not enough players for equal chart")

# ===== Chart 3: Best in Disadvantage (GREEN) =====
print("\nGenerating Chart 3: Best in Disadvantage...")
df_disadv = df_filtered[df_filtered["Disadv_Rounds"] >= MIN_CONDITION_ROUNDS].copy()
print(f"  Players with >={MIN_CONDITION_ROUNDS} disadvantage rounds: {len(df_disadv)}")

if len(df_disadv) >= 10:
    top_disadv = df_disadv.nlargest(10, "Disadv_KD").reset_index(drop=True)
    create_weapon_adv_chart(
        top_disadv,
        "TOP 10 K/D - DISADVANTAGE $",
        "Players with highest K/D when their team has economy disadvantage (±$2000 threshold)",
        output_disadvantage,
        "Disadv_KD",
        DISADV_BAR,
        DISADV_BORDER
    )
else:
    print(f"  [SKIP] Not enough players for disadvantage chart")

# ===== Chart 4: Most Consistent (PURPLE) =====
print("\nGenerating Chart 4: Most Consistent...")
# Need significant rounds in ALL conditions
df_consistent = df_filtered[
    (df_filtered["Adv_Rounds"] >= MIN_CONDITION_ROUNDS) &
    (df_filtered["Equal_Rounds"] >= MIN_CONDITION_ROUNDS) &
    (df_filtered["Disadv_Rounds"] >= MIN_CONDITION_ROUNDS)
].copy()
print(f"  Players with >={MIN_CONDITION_ROUNDS} rounds in ALL conditions: {len(df_consistent)}")

if len(df_consistent) >= 10:
    # Sort by LOWEST variance (most consistent) with high overall K/D as tiebreaker
    top_consistent = df_consistent.nsmallest(10, "KD_Variance").reset_index(drop=True)
    
    # For display, we'll use Overall_KD as the bar metric
    create_weapon_adv_chart(
        top_consistent,
        "TOP 10 MOST CONSISTENT PLAYERS",
        "Players with smallest K/D variance across all economy conditions",
        output_consistent,
        "Overall_KD",
        CONSISTENT_BAR,
        CONSISTENT_BORDER
    )
else:
    print(f"  [SKIP] Not enough players for consistency chart")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"Best in Advantage: {output_advantage}")
print(f"Best in Equal: {output_equal}")
print(f"Best in Disadvantage: {output_disadvantage}")
print(f"Most Consistent: {output_consistent}")