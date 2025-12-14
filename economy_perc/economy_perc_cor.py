import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# === Configuration ===
economy_csv = "weapon_economy_percentage.csv"
performance_csv = "player_performance.csv"  # You need to create this
output_dir = Path("correlation_analysis")
output_dir.mkdir(exist_ok=True)

# === MANUAL CONFIGURATION: Define which players to analyze ===
players_to_analyze = ["m0NESY", "NiKo", "kyousuke"]

# === Functions ===
def parse_placement(placement_str):
    """
    Convert placement string to numeric value.
    Examples: "1" -> 1, "3-4" -> 3.5, "4-8" -> 6, "9-12" -> 10.5
    """
    placement_str = str(placement_str).strip()
    
    if "-" in placement_str:
        # It's a range
        parts = placement_str.split("-")
        try:
            start = float(parts[0])
            end = float(parts[1])
            return (start + end) / 2
        except:
            return None
    else:
        # It's a single number
        try:
            return float(placement_str)
        except:
            return None

def invert_placement(placement_value):
    """
    Invert placement so higher is better.
    1st place (1) -> 1.0, 2nd place (2) -> 0.5, etc.
    """
    if placement_value is None or placement_value == 0:
        return None
    return 1.0 / placement_value

def create_regression_plot(x, y, x_label, y_label, title, output_file, player_name):
    """
    Create scatter plot with regression line and statistics.
    """
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        print(f"Warning: Not enough data points for {player_name} - {title}")
        return
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2
    
    # Create regression line
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = slope * x_line + intercept
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(x_clean, y_clean, alpha=0.6, s=100, edgecolors='black', linewidth=1.5)
    
    # Regression line
    ax.plot(x_line, y_line, 'r-', linewidth=2, label='Regression Line')
    
    # Add statistics text
    stats_text = f'R² = {r_squared:.4f}\n'
    stats_text += f'p-value = {p_value:.4f}\n'
    stats_text += f'y = {slope:.4f}x + {intercept:.4f}'
    
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(f'{player_name}: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return r_squared, p_value, slope, intercept

# === Load data ===
print("Loading data...")
economy_df = pd.read_csv(economy_csv)
economy_df["Event"] = economy_df["Event"].str.strip()

try:
    performance_df = pd.read_csv(performance_csv)
    performance_df["Event"] = performance_df["Event"].str.strip()
except FileNotFoundError:
    print(f"Error: {performance_csv} not found!")
    print("Please create a CSV file with columns: Player, Event, Rating, Placement")
    exit()

# Validate performance_df columns
required_cols = ["Player", "Event", "Rating", "Placement"]
if not all(col in performance_df.columns for col in required_cols):
    print(f"Error: {performance_csv} must have columns: {required_cols}")
    exit()

# === Process placement data ===
print("Processing placement data...")
performance_df["PlacementValue"] = performance_df["Placement"].apply(parse_placement)
performance_df["InvertedPlacement"] = performance_df["PlacementValue"].apply(invert_placement)

# === Analysis ===
print("\n" + "="*80)
print("CORRELATION ANALYSIS RESULTS")
print("="*80)

for player in players_to_analyze:
    print(f"\n{'='*80}")
    print(f"Analyzing: {player}")
    print(f"{'='*80}")
    
    # Filter data for this player
    player_economy = economy_df[
        (economy_df["Player"] == player) & 
        (economy_df["Event"] != "overall")
    ].copy()
    
    player_performance = performance_df[performance_df["Player"] == player].copy()
    
    if player_economy.empty:
        print(f"Warning: No economy data found for {player}")
        continue
    
    if player_performance.empty:
        print(f"Warning: No performance data found for {player}")
        continue
    
    # Merge data
    merged_data = player_economy.merge(
        player_performance,
        on=["Player", "Event"],
        how="inner"
    )
    
    if merged_data.empty:
        print(f"Warning: No matching events found for {player}")
        continue
    
    print(f"\nEvents analyzed: {len(merged_data)}")
    print(f"Events: {', '.join(merged_data['Event'].tolist())}")
    
    # Extract data for analysis
    economy_pct = merged_data["AvgPercentageOfTeam"].values
    rating = merged_data["Rating"].values
    inverted_placement = merged_data["InvertedPlacement"].values
    
    # === Economy vs Rating ===
    print(f"\n--- Economy % vs Rating ---")
    output_file_rating = output_dir / f"{player}_economy_vs_rating.png"
    
    try:
        r2_rating, p_rating, slope_rating, intercept_rating = create_regression_plot(
            economy_pct,
            rating,
            "Average % of Team Weapon Value",
            "Rating",
            "Economy Percentage vs Rating",
            output_file_rating,
            player
        )
        
        print(f"R² = {r2_rating:.4f}")
        print(f"P-value = {p_rating:.4f}")
        print(f"Equation: Rating = {slope_rating:.4f} × Economy% + {intercept_rating:.4f}")
        print(f"Saved plot: {output_file_rating}")
        
        if p_rating < 0.05:
            print("✓ Statistically significant (p < 0.05)")
        else:
            print("✗ Not statistically significant (p >= 0.05)")
    
    except Exception as e:
        print(f"Error creating rating plot: {e}")
    
    # === Economy vs Placement ===
    print(f"\n--- Economy % vs Team Placement ---")
    output_file_placement = output_dir / f"{player}_economy_vs_placement.png"
    
    # Remove rows with NaN inverted placement
    mask = ~np.isnan(inverted_placement)
    economy_pct_clean = economy_pct[mask]
    inverted_placement_clean = inverted_placement[mask]
    
    if len(economy_pct_clean) < 3:
        print(f"Warning: Not enough valid placement data for {player}")
    else:
        try:
            r2_placement, p_placement, slope_placement, intercept_placement = create_regression_plot(
                economy_pct_clean,
                inverted_placement_clean,
                "Average % of Team Weapon Value",
                "Inverted Placement Score (Higher = Better)",
                "Economy Percentage vs Team Placement",
                output_file_placement,
                player
            )
            
            print(f"R² = {r2_placement:.4f}")
            print(f"P-value = {p_placement:.4f}")
            print(f"Equation: Placement = {slope_placement:.4f} × Economy% + {intercept_placement:.4f}")
            print(f"Saved plot: {output_file_placement}")
            
            if p_placement < 0.05:
                print("✓ Statistically significant (p < 0.05)")
            else:
                print("✗ Not statistically significant (p >= 0.05)")
        
        except Exception as e:
            print(f"Error creating placement plot: {e}")

print(f"\n{'='*80}")
print("Analysis complete! Check the '{output_dir}' folder for plots.")
print(f"{'='*80}")