"""Temporary debug script -- dumps the full feature vector for one fight
so it can be diffed across environments (local vs CI)."""
import sys, json
sys.path.insert(0, ".")
import predict as P
from config import DB_V1_PATH, MODELS_V1_PROD_DIR

_orig = P.build_feature_vector
_captured = []

def _wrapped(*args, **kwargs):
    df = _orig(*args, **kwargs)
    _captured.append(df)
    return df

P.build_feature_vector = _wrapped

result = P.compute_prediction(
    red_name="Uros Medic",
    blue_name="Daniel Rodriguez",
    model_type="ensemble",
    division="welterweight",
    title_fight=0,
    db_path=DB_V1_PATH,
    models_dir=MODELS_V1_PROD_DIR,
    r_fighter_id="681399317dbf4701",
    b_fighter_id="8a1f3b5c526cd6e6",
)

print("=== RESULT ===")
print("red_prob:", result["red_prob"], "blue_prob:", result["blue_prob"])

# Print the union of all feature vectors captured (one per base model, should
# all carry the same values for shared feature names)
print("=== FEATURES (first captured vector, full) ===")
if _captured:
    df = _captured[0]
    row = df.iloc[0].to_dict()
    for k in sorted(row.keys()):
        print(f"{k} = {row[k]}")
print(f"=== {len(_captured)} feature vectors captured total ===")
