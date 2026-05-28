import pandas as pd
import sqlite3
from pathlib import Path

# ADJUST PATH IF NEEDED
CURRENT_DIR = Path(__file__).resolve().parent
DB_PATH = CURRENT_DIR.parent / "database_builder_files" / "ufc_v2.db"


def inspect_leaderboard():
    # 1. Run the Elo Calculation Logic (Copy-Paste basics or import)
    # Ideally, import your existing function. Here I recreate the minimal check.
    # Note: We load the CSV you already made to save time if possible,
    # but it's safer to recalculate to see the raw state.

    conn = sqlite3.connect(str(DB_PATH))
    print("Loading data...")
    query = """
    SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id
    FROM fights 
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Elo Params
    STARTING_ELO = 1400
    K_FACTOR = 60  # Using a high K for this test to see if movement happens

    fighter_ratings = {}
    fighter_names = {}  # Map ID to Name

    print(f"Processing {len(df)} fights...")
    for index, row in df.iterrows():
        r_id = row['r_fighter_id']
        b_id = row['b_fighter_id']
        winner = row['winner_id']

        # Map Names
        fighter_names[r_id] = row['r_fighter']
        fighter_names[b_id] = row['b_fighter']

        r_curr = fighter_ratings.get(r_id, STARTING_ELO)
        b_curr = fighter_ratings.get(b_id, STARTING_ELO)

        # Calc Expected
        exp_r = 1 / (1 + 10 ** ((b_curr - r_curr) / 400))

        # Score
        if winner == r_id:
            score_r = 1.0
        elif winner == b_id:
            score_r = 0.0
        else:
            score_r = 0.5

        # Update
        new_r = r_curr + K_FACTOR * (score_r - exp_r)
        new_b = b_curr + K_FACTOR * ((1 - score_r) - (1 - exp_r))

        fighter_ratings[r_id] = new_r
        fighter_ratings[b_id] = new_b

    # Convert to DataFrame for sorting
    leaderboard = pd.DataFrame(list(fighter_ratings.items()), columns=['id', 'elo'])
    leaderboard['name'] = leaderboard['id'].map(fighter_names)

    # Sort by Top Elo
    top_10 = leaderboard.sort_values('elo', ascending=False).head(15)

    print("\n=== TOP 15 FIGHTERS BY ELO (SANITY CHECK) ===")
    print(top_10[['name', 'elo']])


if __name__ == "__main__":
    inspect_leaderboard()