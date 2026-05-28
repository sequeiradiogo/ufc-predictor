import sqlite3
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

DB_PATH = CURRENT_DIR.parent / "database_builder_files" / "ufc_v2.db"


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA table_info(fight_stats);")
#for row in cur.fetchall():
 #   print(row)



query2 = """
SELECT count(*) 
FROM fight_stats AS fs
JOIN fights AS f ON fs.fight_id = f.fight_id
JOIN fighters AS fi ON fi.fighter_id = fs.fighter_id
WHERE fs.total_fight_time = 0
ORDER BY f.date DESC
"""
values = cur.execute(query2).fetchall()

for v in values:
    print(v)
    pass
