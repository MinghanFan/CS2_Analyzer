import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("post_visualizations")
output_dir.mkdir(exist_ok=True)

# Use similar styling to exit frag charts
plt.rcParams['font.family'] = 'sans-serif'

print("Creating Social Media Infographic...")

fig = plt.figure(figsize=(11, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title (similar style to exit frag charts)
ax.text(50, 93, 'FALCONS ECONOMY EFFICIENCY', 
        ha='center', va='center',
        fontsize=26, fontweight='bold', color='#4a545b')

ax.text(50, 89.6, 'Who Converts Resources to Wins?', 
        ha='center', va='center',
        fontsize=14, style='italic', color='#7f8c8d')

# Bar chart section
players = ['m0NESY', 'NiKo', 'kyousuke']
avg_coefs = [0.217, 0.083, -0.156]
colors = ['#8bd0a7', '#f39c12', '#e17c7c']  # Green, yellow, red
edge_colors = ['#8ab7a0', '#d68910', '#c47d7d']

# Bar positions (tighter spacing)
bar_y_start = 80
bar_spacing = 12
bar_height = 10

for idx, (player, coef, color, edge) in enumerate(zip(players, avg_coefs, colors, edge_colors)):
    y_pos = bar_y_start - (idx * bar_spacing)
    
    # Add light grey background for alternating rows
    if idx % 2 == 0:
        grey_rect = plt.Rectangle((5, y_pos - bar_height/2 - 0.5), 90, bar_height + 1,
                                 facecolor='#f2f2f2', edgecolor='none',
                                 zorder=0, transform=ax.transData)
        ax.add_patch(grey_rect)
    
    # Player name
    ax.text(8, y_pos, player, 
            va='center', ha='left',
            fontsize=16, fontweight='bold', color='#4a545b')
    
    # Bar - zero line is at center
    zero_line_x = 50  # Center position for zero
    max_bar_width = 30  # Max width in each direction from zero
    
    # Normalize bar width (scale to max coefficient)
    max_abs_coef = max(abs(c) for c in avg_coefs)
    bar_width = (abs(coef) / max_abs_coef) * max_bar_width
    
    # Draw bar - negative goes left, positive goes right
    if coef >= 0:
        # Positive: bar goes right from zero line
        bar_rect = plt.Rectangle((zero_line_x, y_pos - bar_height/2 + 2), bar_width, bar_height - 4,
                                 facecolor=color, edgecolor=edge, 
                                 linewidth=2.5, transform=ax.transData)
        value_x = zero_line_x + bar_width + 2
        value_ha = 'left'
    else:
        # Negative: bar goes left from zero line
        bar_rect = plt.Rectangle((zero_line_x - bar_width, y_pos - bar_height/2 + 2), bar_width, bar_height - 4,
                                 facecolor=color, edgecolor=edge, 
                                 linewidth=2.5, transform=ax.transData)
        value_x = zero_line_x - bar_width - 2
        value_ha = 'right'
    
    ax.add_patch(bar_rect)
    
    # Coefficient value at end of bar
    ax.text(value_x, y_pos, f'{coef:+.3f}',
            va='center', ha=value_ha,
            fontsize=13, fontweight='bold', color='#4b5964')

# Zero line (centered)
zero_line_x = 50
ax.plot([zero_line_x, zero_line_x], [bar_y_start + 8, bar_y_start - (len(players)-1)*bar_spacing - 8],
        color='#4a545b', linewidth=2, zorder=1)

# Interpretation (below bars, closer)
interp_y = bar_y_start - (len(players) * bar_spacing) +3
ax.text(50, interp_y, 
        'Positive = Higher economy share correlates with better team placement\n' +
        'Negative = Higher economy share correlates with worse team placement',
        ha='center', va='top',
        fontsize=10, color='#7f8c8d',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.7, edgecolor='#bdbdbd'))

# Disclaimer (similar to exit frag charts)
ax.text(50, 41, 
        'Limitations: Small sample (6-9 events) | No statistical significance (p>0.05) | Exploratory analysis only',
        ha='center', va='bottom',
        fontsize=8, style='italic', color='#9aa3aa')

ax.text(50, 39, 
        'Data: 2025 Big Event events since Melbourne | Method: Additive regression models | Metric: Economy % to Inverted Placement | By clu0ki @creniusz',
        ha='center', va='bottom',
        fontsize=7, color='#9aa3aa')

plt.tight_layout(pad=0)
plt.savefig(output_dir / 'social_media_infographic.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: social_media_infographic.png")
plt.close()

print(f"\nVisualization saved to '{output_dir}/' folder!")
print("\nSUGGESTED POST CAPTION:")
print("""
Falcons Economy Analysis: Resource Efficiency Study

Analyzed how weapon economy distribution relates to team performance across 2025 events.

Key Pattern: m0NESY shows strongest positive correlation (+0.22 avg) - when he gets more 
resources, team tends to place better.

Important: Small sample size (6-9 events), not statistically significant. This is 
exploratory analysis showing patterns, not proof.

#CS2Stats #Falcons #DataAnalysis #Esports
""")