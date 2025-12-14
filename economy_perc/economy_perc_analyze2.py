import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
performance_csv = "player_performance.csv"
output_base_dir = Path("rating_correlation_analysis")
output_base_dir.mkdir(exist_ok=True)

# === MANUAL CONFIGURATION: Define team combinations to analyze ===
team_combinations = {
    "Falcons_Core": ["m0NESY", "NiKo", "kyousuke"],
    "Falcons_Duo_mNiko": ["m0NESY", "NiKo"],
    "Falcons_Duo_mKyou": ["m0NESY", "kyousuke"],
    "Falcons_Duo_NikoKyou": ["NiKo", "kyousuke"]
}

# === Functions ===
def parse_placement(placement_str):
    """Convert placement string to numeric value."""
    placement_str = str(placement_str).strip()
    if "-" in placement_str:
        parts = placement_str.split("-")
        try:
            start = float(parts[0])
            end = float(parts[1])
            return (start + end) / 2
        except:
            return None
    else:
        try:
            return float(placement_str)
        except:
            return None

def invert_placement(placement_value):
    """Invert placement so higher is better."""
    if placement_value is None or placement_value == 0:
        return None
    try:
        return 1.0/(placement_value**1.2)
    except Exception:
        return None

def fit_additive_model(X, y, player_names):
    """Fit additive model: y = b0 + b1*x1 + b2*x2 + ..."""
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    
    # Calculate adjusted R²
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    coefficients = {}
    for i, name in enumerate(player_names):
        coefficients[name] = model.coef_[i]
    coefficients['intercept'] = model.intercept_
    
    return {
        'model': model,
        'r2': r2,
        'adj_r2': adj_r2,
        'coefficients': coefficients,
        'formula': ' + '.join([f"{coefficients[name]:.4f}*{name}" for name in player_names])
    }

def fit_multiplicative_model(X, y, player_names):
    """Fit multiplicative model: y = b0 + b1*(x1*x2*...)"""
    X_mult = np.prod(X, axis=1).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_mult, y)
    r2 = r2_score(y, model.predict(X_mult))
    
    n = len(y)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    coefficients = {
        'product': model.coef_[0],
        'intercept': model.intercept_
    }
    
    product_term = '*'.join(player_names)
    
    return {
        'model': model,
        'r2': r2,
        'adj_r2': adj_r2,
        'coefficients': coefficients,
        'formula': f"{coefficients['product']:.4f}*({product_term})",
        'X_transformed': X_mult
    }

def fit_full_interaction_model(X, y, player_names):
    """Fit model with all interactions (pairwise + higher-order)."""
    n_players = len(player_names)
    
    # Start with individual terms
    X_full = X.copy()
    feature_names = player_names.copy()
    
    # Add pairwise interactions
    if n_players >= 2:
        for i in range(n_players):
            for j in range(i+1, n_players):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                X_full = np.hstack([X_full, interaction])
                feature_names.append(f"{player_names[i]}*{player_names[j]}")
    
    # Add 3-way interaction for trios
    if n_players == 3:
        interaction_3way = (X[:, 0] * X[:, 1] * X[:, 2]).reshape(-1, 1)
        X_full = np.hstack([X_full, interaction_3way])
        feature_names.append(f"{player_names[0]}*{player_names[1]}*{player_names[2]}")
    
    model = LinearRegression()
    model.fit(X_full, y)
    r2 = r2_score(y, model.predict(X_full))
    
    n = len(y)
    p = X_full.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    coefficients = {}
    for i, name in enumerate(feature_names):
        coefficients[name] = model.coef_[i]
    coefficients['intercept'] = model.intercept_
    
    return {
        'model': model,
        'r2': r2,
        'adj_r2': adj_r2,
        'coefficients': coefficients,
        'feature_names': feature_names,
        'X_transformed': X_full
    }

def fit_simplified_interaction_model(X, y, player_names):
    """Fit model with individual terms + highest-order interaction only."""
    n_players = len(player_names)
    
    # Individual terms + product of all
    X_simple = X.copy()
    feature_names = player_names.copy()
    
    if n_players >= 2:
        interaction_full = np.prod(X, axis=1).reshape(-1, 1)
        X_simple = np.hstack([X_simple, interaction_full])
        feature_names.append('*'.join(player_names))
    
    model = LinearRegression()
    model.fit(X_simple, y)
    r2 = r2_score(y, model.predict(X_simple))
    
    n = len(y)
    p = X_simple.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    coefficients = {}
    for i, name in enumerate(feature_names):
        coefficients[name] = model.coef_[i]
    coefficients['intercept'] = model.intercept_
    
    return {
        'model': model,
        'r2': r2,
        'adj_r2': adj_r2,
        'coefficients': coefficients,
        'feature_names': feature_names,
        'X_transformed': X_simple
    }

def plot_model_comparison(results, output_file, combination_name):
    """Bar chart comparing R² across models."""
    models = list(results.keys())
    r2_values = [results[m]['r2'] for m in models]
    adj_r2_values = [results[m]['adj_r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, r2_values, width, label='R²', alpha=0.8)
    bars2 = ax.bar(x + width/2, adj_r2_values, width, label='Adjusted R²', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{combination_name}: Model Comparison (Rating)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_coefficients(coefficients_dict, output_file, combination_name, model_name):
    """Bar chart showing coefficients (positive/negative)."""
    # Filter out intercept
    coefs = {k: v for k, v in coefficients_dict.items() if k != 'intercept'}
    
    if not coefs:
        return
    
    names = list(coefs.keys())
    values = list(coefs.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Term', fontsize=12, fontweight='bold')
    ax.set_title(f'{combination_name} - {model_name}: Coefficients (Rating)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        label_x = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(label_x, i, f'{val:.4f}', va='center', ha=ha, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_surface(X, y, player_names, model_result, output_file, combination_name):
    """3D surface plot for duo analysis."""
    if X.shape[1] != 2:
        return
    
    # Create mesh grid
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    # Predict on mesh
    X_mesh = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
    y_pred = model_result['model'].predict(X_mesh)
    y_mesh = y_pred.reshape(x1_mesh.shape)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface
    surf = ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.6, cmap='viridis')
    
    # Actual data points
    ax.scatter(X[:, 0], X[:, 1], y, c='red', marker='o', s=100, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel(f'{player_names[0]} Rating', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'{player_names[1]} Rating', fontsize=10, fontweight='bold')
    ax.set_zlabel('Inverted Placement', fontsize=10, fontweight='bold')
    ax.set_title(f'{combination_name}: 3D Surface (Rating)', fontsize=12, fontweight='bold')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_contour(X, y, player_names, model_result, output_file, combination_name):
    """Contour plot for duo analysis."""
    if X.shape[1] != 2:
        return
    
    # Create mesh grid
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    # Predict on mesh
    X_mesh = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
    y_pred = model_result['model'].predict(X_mesh)
    y_mesh = y_pred.reshape(x1_mesh.shape)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(x1_mesh, x2_mesh, y_mesh, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax.contour(x1_mesh, x2_mesh, y_mesh, levels=10, colors='black', linewidths=0.5, alpha=0.4)
    
    # Actual data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='black', linewidth=1.5, cmap='coolwarm', zorder=5)
    
    ax.set_xlabel(f'{player_names[0]} Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{player_names[1]} Rating', fontsize=12, fontweight='bold')
    ax.set_title(f'{combination_name}: Contour Plot (Rating)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Inverted Placement', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_partial_dependence(X, y, player_names, model_result, output_file, combination_name):
    """Partial dependence plots showing each player's effect."""
    n_players = len(player_names)
    
    fig, axes = plt.subplots(1, n_players, figsize=(6*n_players, 5))
    if n_players == 1:
        axes = [axes]
    
    for idx, (ax, player_name) in enumerate(zip(axes, player_names)):
        # Create range for this player
        x_range = np.linspace(X[:, idx].min(), X[:, idx].max(), 100)
        
        # Hold other players at their mean
        X_partial = np.tile(X.mean(axis=0), (100, 1))
        X_partial[:, idx] = x_range
        
        # Predict
        y_pred = model_result['model'].predict(X_partial)
        
        # Plot
        ax.plot(x_range, y_pred, linewidth=2, color='blue')
        ax.scatter(X[:, idx], y, alpha=0.5, s=50, edgecolors='black')
        
        ax.set_xlabel(f'{player_name} Rating', fontsize=11, fontweight='bold')
        ax.set_ylabel('Inverted Placement', fontsize=11, fontweight='bold')
        ax.set_title(f'Effect of {player_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{combination_name}: Partial Dependence (Rating)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# === Load data ===
print("Loading data...")
try:
    performance_df = pd.read_csv(performance_csv)
    performance_df["Event"] = performance_df["Event"].str.strip()
except FileNotFoundError:
    print(f"Error: {performance_csv} not found!")
    exit()

# Process placement data
performance_df["PlacementValue"] = performance_df["Placement"].apply(parse_placement)
performance_df["InvertedPlacement"] = performance_df["PlacementValue"].apply(invert_placement)

# === Main Analysis Loop ===
print("\n" + "="*100)
print("MULTI-PLAYER CORRELATION ANALYSIS (RATING)")
print("="*100)

for combination_name, players in team_combinations.items():
    print(f"\n{'='*100}")
    print(f"Analyzing: {combination_name} - {players}")
    print(f"{'='*100}")
    
    # Create output directory for this combination
    output_dir = output_base_dir / combination_name
    output_dir.mkdir(exist_ok=True)
    
    # Gather data for all players in this combination
    combined_data = []
    
    for player in players:
        player_perf = performance_df[
            performance_df["Player"] == player
        ][["Player", "Event", "Rating", "InvertedPlacement"]].copy()
        
        combined_data.append(player_perf)
    
    if not combined_data:
        print(f"Warning: No data found for any players in {combination_name}")
        continue
    
    # Find common events across all players
    common_events = set(combined_data[0]["Event"].unique())
    for df in combined_data[1:]:
        common_events &= set(df["Event"].unique())
    
    if not common_events:
        print(f"Warning: No common events found for all players in {combination_name}")
        continue
    
    print(f"\nCommon events: {len(common_events)}")
    print(f"Events: {', '.join(sorted(common_events))}")
    
    # Build feature matrix
    analysis_data = []
    for event in sorted(common_events):
        row = {"Event": event}
        for player in players:
            player_data = combined_data[players.index(player)]
            event_data = player_data[player_data["Event"] == event]
            if not event_data.empty:
                row[f"{player}_Rating"] = event_data["Rating"].values[0]
                row["InvertedPlacement"] = event_data["InvertedPlacement"].values[0]
        
        if len(row) == len(players) + 2:  # All players + Event + Placement
            analysis_data.append(row)
    
    if len(analysis_data) < 3:
        print(f"Warning: Not enough data points ({len(analysis_data)}) for regression")
        continue
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Prepare X and y
    X = analysis_df[[f"{player}_Rating" for player in players]].values
    y = analysis_df["InvertedPlacement"].values
    
    print(f"\nData points for analysis: {len(analysis_df)}")
    
    # === Fit all models ===
    results = {}
    
    print("\n--- Fitting Models ---")
    
    # Model 1: Additive
    print("Fitting Additive Model...")
    results['Additive'] = fit_additive_model(X, y, players)
    
    # Model 2: Multiplicative
    print("Fitting Multiplicative Model...")
    results['Multiplicative'] = fit_multiplicative_model(X, y, players)
    
    # Model 3: Full Interactions
    print("Fitting Full Interactions Model...")
    results['Full_Interactions'] = fit_full_interaction_model(X, y, players)
    
    # Model 4: Simplified Interactions
    print("Fitting Simplified Interactions Model...")
    results['Simplified_Interactions'] = fit_simplified_interaction_model(X, y, players)
    
    # === Save Results ===
    
    # Model comparison CSV
    comparison_rows = []
    for model_name, result in results.items():
        comparison_rows.append({
            'Model': model_name,
            'R²': result['r2'],
            'Adjusted_R²': result['adj_r2'],
            'Formula': result.get('formula', 'See coefficients')
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / f"{combination_name}_model_comparison.csv", index=False)
    
    # Coefficients CSV
    coef_rows = []
    for model_name, result in results.items():
        for term, value in result['coefficients'].items():
            interpretation = ""
            if term != 'intercept' and term != 'product':
                if value > 0:
                    interpretation = "↑ (Higher rating → Better placement)"
                else:
                    interpretation = "↓ (Higher rating → Worse placement)"
            
            coef_rows.append({
                'Model': model_name,
                'Term': term,
                'Coefficient': value,
                'Interpretation': interpretation
            })
    
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(output_dir / f"{combination_name}_coefficients.csv", index=False)
    
    # === Generate Plots ===
    
    print("\n--- Generating Visualizations ---")
    
    # Model comparison plot
    plot_model_comparison(
        results,
        output_dir / f"{combination_name}_model_r2_comparison.png",
        combination_name
    )
    print(f"✓ Saved: model_r2_comparison.png")
    
    # Coefficient plots for each model
    for model_name, result in results.items():
        plot_coefficients(
            result['coefficients'],
            output_dir / f"{combination_name}_{model_name}_coefficients.png",
            combination_name,
            model_name
        )
        print(f"✓ Saved: {model_name}_coefficients.png")
    
    # Duo-specific plots
    if len(players) == 2:
        # Use additive model for visualization
        plot_3d_surface(
            X, y, players, results['Additive'],
            output_dir / f"{combination_name}_3d_surface.png",
            combination_name
        )
        print(f"✓ Saved: 3d_surface.png")
        
        plot_contour(
            X, y, players, results['Additive'],
            output_dir / f"{combination_name}_contour_plot.png",
            combination_name
        )
        print(f"✓ Saved: contour_plot.png")
    
    # Partial dependence plots
    plot_partial_dependence(
        X, y, players, results['Additive'],
        output_dir / f"{combination_name}_partial_dependence.png",
        combination_name
    )
    print(f"✓ Saved: partial_dependence.png")
    
    # === Print interpretations ===
    print("\n--- Model Results Summary ---")
    
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
    
    print("\n--- Additive Model Interpretation ---")
    for player in players:
        coef = results['Additive']['coefficients'][player]
        if coef > 0:
            print(f"  {player}: +{coef:.4f} → Higher rating → Better placement ✓")
        else:
            print(f"  {player}: {coef:.4f} → Higher rating → Worse placement ✗")
    
    if 'Full_Interactions' in results:
        print("\n--- Full Interactions Model ---")
        for term, coef in results['Full_Interactions']['coefficients'].items():
            if term != 'intercept':
                direction = "positive" if coef > 0 else "negative"
                print(f"  {term}: {coef:.4f} ({direction} effect)")

print(f"\n{'='*100}")
print(f"Analysis complete! Check the '{output_base_dir}' folder for all results.")
print(f"{'='*100}")