
import ast
import pandas as pd

root = "data/raw/ptb-xl"

df = pd.read_csv(f"{root}/ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(f"{root}/scp_statements.csv", index_col=0)

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

diagnostic = scp[
    (pd.to_numeric(scp["diagnostic"], errors="coerce") == 1)
    & scp["diagnostic_class"].isin(["NORM", "MI", "STTC", "CD", "HYP"])
]

mapping = diagnostic["diagnostic_class"].to_dict()

def get_classes(codes):
    return sorted({
        mapping[c]
        for c, likelihood in codes.items()
        if c in mapping and likelihood > 0
    })

df["classes"] = df["scp_codes"].apply(get_classes)

conflicts = df[
    df["classes"].apply(
        lambda labels: "NORM" in labels
        and any(c != "NORM" for c in labels)
    )
]

print("Total conflicts:", len(conflicts))
print("\nFirst 15 conflicting records:\n")

for ecg_id, row in conflicts.head(15).iterrows():
    print("ECG ID:", ecg_id)
    print("SCP codes:", row["scp_codes"])
    print("Mapped classes:", row["classes"])
    print("-" * 50)
