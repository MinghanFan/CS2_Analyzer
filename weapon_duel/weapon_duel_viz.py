import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

# ===== Configuration =====
csv_file = "weapon_duel_economy_analysis.csv"
player_photos_dir = Path("player_photos")
output_higher = "weapon_duel_higher_econ_ex.png"
output_equal = "weapon_duel_equal_econ_ex.png"
output_lower = "weapon_duel_lower_econ_ex.png"
MIN_ROUNDS = 1200

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
    "XANTARES": "Aurora",
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
    "Wicadia": "Aurora",
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
    "woxic": "Aurora",
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
    "jottAAA": "Aurora",
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
    "MAJ3R": "Aurora",
    "cadiaN": "Astralis",
}

# ===== ABSOLUTE POSITION CONTROLS =====
FIGURE_WIDTH_INCHES = 11
FIGURE_HEIGHT_INCHES = 10

PHOTO_X = 8
NAME_X = 14.5
BAR_START_X = 32-3
BAR_END_X = 75-3
STATS_X_OFFSET = 2

Y_START = 12
Y_SPACING = 9

PHOTO_SIZE = 57
PHOTO_BOX_SIZE = 9
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

# ===== COLORS BY ECONOMY CONDITION =====
HIGHER_BAR = "#dda19e"
HIGHER_BORDER = "#c58b7d"

EQUAL_BAR = "#7bb3e0"
EQUAL_BORDER = "#6a9dca"

LOWER_BAR = "#8bd0a7"
LOWER_BORDER = "#8ab7a0"

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

def create_weapon_duel_chart(data, title, subtitle, output_file, metric_col, bar_color, border_color):
    """Create chart with ABSOLUTE positioning - matching weapon advantage style"""
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    kd_values = [row[metric_col] for _, row in data.iterrows()]
    max_kd = max(kd_values)
    
    for idx, (i, row) in enumerate(data.iterrows()):
        player_name = row["Player"]
        
        higher_kd = row["Higher_KD"]
        equal_kd = row["Equal_KD"]
        lower_kd = row["Lower_KD"]
        
        primary_kd = row[metric_col]
        
        y_pos = 100 - Y_START - (idx * Y_SPACING)
        
        # Add grey background for even rows
        if idx % 2 == 0:
            grey_rect = plt.Rectangle((0, y_pos - 4.475), 107, BAR_HEIGHT*5.29,
                                     facecolor="#f2f2f2", edgecolor='none',
                                     zorder=0, transform=ax.transData, clip_on=False)
            ax.add_patch(grey_rect)
        
        # Get team color
        team_name = PLAYER_TEAMS.get(player_name, "default")
        team_color = TEAM_COLORS.get(team_name, TEAM_COLORS["default"])
        
        # Draw FIXED-SIZE background square for photo
        photo_bg = plt.Rectangle(
            (PHOTO_X - PHOTO_BOX_SIZE/2, y_pos - PHOTO_BOX_SIZE/2),
            PHOTO_BOX_SIZE,
            PHOTO_BOX_SIZE,
            facecolor=team_color,
            edgecolor=border_color,
            linewidth=PHOTO_BORDER,
            transform=ax.transData,
            zorder=1
        )
        ax.add_patch(photo_bg)
        
        # Add photo on top of background square (no frame)
        img = load_player_image(player_name, player_photos_dir, size=PHOTO_SIZE)
        imagebox = AnnotationBbox(img, (PHOTO_X, y_pos), 
                                 frameon=False,
                                 box_alignment=(0.5, 0.5),
                                 zorder=2)
        ax.add_artist(imagebox)
        
        # Add player name
        ax.text(NAME_X, y_pos, player_name, 
               va='center', ha='left', 
               fontsize=NAME_FONT, fontweight='bold',
               color='#4a545b', transform=ax.transData)
        
        # Add bar
        bar_length = (primary_kd / max_kd) * (BAR_END_X - BAR_START_X)
        bar_rect = plt.Rectangle((BAR_START_X, y_pos - BAR_HEIGHT/2), bar_length, BAR_HEIGHT,
                                 facecolor=bar_color, edgecolor=border_color, 
                                 linewidth=BAR_BORDER, transform=ax.transData)
        ax.add_patch(bar_rect)
        
        # Add stats
        stats_x = BAR_START_X + bar_length + STATS_X_OFFSET
        stats_text = f"Higher: {higher_kd:.2f}  |  Equal: {equal_kd:.2f}  |  Lower: {lower_kd:.2f}"
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
    ax.text(102.75, 2, f'Minimum {MIN_ROUNDS} rounds | Excluding AWP duels',
           ha='right', va='bottom',
           fontsize=8, style='italic', color='#9aa3aa',
           transform=ax.transData)
    
    # Add credits
    ax.text(102.75, 0.5, 'Photo by HLTV | Data by clu0ki',
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

# Filter for include_awp rows only
# df = df[df["AWP_Filter"] == "include_awp"].copy()
# print(f"Players (include_awp): {len(df)}")

# Filter for exclude_awp rows only
df = df[df["AWP_Filter"] == "exclude_awp"].copy()
print(f"Players (exclude_awp): {len(df)}")

# Calculate K/D ratios
df["Higher_KD"] = df["Higher_Econ_Kills"] / df["Higher_Econ_Deaths"].replace(0, 1)
df["Equal_KD"] = df["Equal_Econ_Kills"] / df["Equal_Econ_Deaths"].replace(0, 1)
df["Lower_KD"] = df["Lower_Econ_Kills"] / df["Lower_Econ_Deaths"].replace(0, 1)

# Filter by minimum rounds
df_filtered = df[df["Total_Rounds"] >= MIN_ROUNDS].copy()
print(f"Players with >={MIN_ROUNDS} rounds: {len(df_filtered)}")

if len(df_filtered) < 10:
    print(f"[ERROR] Not enough players with {MIN_ROUNDS} rounds")
    exit(1)

# Chart 1: Best Higher Economy K/D
print("\nGenerating Chart 1: Best Higher Economy K/D...")
top_higher = df_filtered.nlargest(10, "Higher_KD").reset_index(drop=True)
create_weapon_duel_chart(
    top_higher,
    "TOP 10 K/D - HIGHER WEAPON $",
    "Players with highest K/D when holding more expensive weapon (±$200 threshold)",
    output_higher,
    "Higher_KD",
    HIGHER_BAR,
    HIGHER_BORDER
)

# Chart 2: Best Equal Economy K/D
print("\nGenerating Chart 2: Best Equal Economy K/D...")
top_equal = df_filtered.nlargest(10, "Equal_KD").reset_index(drop=True)
create_weapon_duel_chart(
    top_equal,
    "TOP 10 K/D - EQUAL WEAPON $",
    "Players with highest K/D in fair weapon duels (±$200 threshold)", # Exclude AWP duels:
    output_equal,
    "Equal_KD",
    EQUAL_BAR,
    EQUAL_BORDER
)

# Chart 3: Best Lower Economy K/D
print("\nGenerating Chart 3: Best Lower Economy K/D...")
top_lower = df_filtered.nlargest(10, "Lower_KD").reset_index(drop=True)
create_weapon_duel_chart(
    top_lower,
    "TOP 10 K/D - LOWER WEAPON $",
    "Players with highest K/D when holding cheaper weapon (±$200 threshold)",
    output_lower,
    "Lower_KD",
    LOWER_BAR,
    LOWER_BORDER
)

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"Higher Economy K/D: {output_higher}")
print(f"Equal Economy K/D: {output_equal}")
print(f"Lower Economy K/D: {output_lower}")