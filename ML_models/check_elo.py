import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
CSV_PATH = "ufc_ml_data_with_debuts_and_elo.csv"  # Or whatever your latest file is called


def audit_elo_feature():
    # 1. Load Data
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # 2. Check if Elo exists
    if 'elo_diff' not in df.columns:
        print("ERROR: 'elo_diff' column not found in CSV.")
        return

    # 3. Create 'Bins' for Elo Difference
    # We group fights by how big the rating gap was.
    # e.g., "Diff between 0-50", "Diff between 50-100"
    df['elo_bin'] = pd.cut(df['elo_diff'], bins=10)

    # 4. Calculate Win Rate per Bin
    # We want to see: As Elo Diff gets higher, does Win % go up?
    audit = df.groupby('elo_bin')[['target']].mean().reset_index()
    audit['count'] = df.groupby('elo_bin')['target'].count().values

    # Filter out empty bins
    audit = audit[audit['count'] > 50]

    print("\n=== ELO AUDIT RESULTS ===")
    print(audit)

    # 5. Plot it
    plt.figure(figsize=(10, 6))
    sns.barplot(data=audit, x='elo_bin', y='target', color='skyblue')
    plt.axhline(0.5, color='red', linestyle='--')
    plt.xticks(rotation=45)
    plt.ylabel("Actual Win Percentage")
    plt.xlabel("Elo Difference (Red - Blue)")
    plt.title("Does Higher Elo actually mean more Wins?")
    plt.tight_layout()
    plt.show()

    # 6. Correlation Check
    corr = df[['target', 'elo_diff']].corr().iloc[0, 1]
    print(f"\nCorrelation between Elo Diff and Target: {corr:.4f}")
    print("(0.0 = No predictive power, 0.2+ = Good, 0.4+ = Excellent)")


if __name__ == "__main__":
    audit_elo_feature()