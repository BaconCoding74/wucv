"""
WuWa Item Dataset Manager
Usage:
    python dataset_manager.py list
    python dataset_manager.py rename <hash> <new_name>
    python dataset_manager.py delete <hash>
    python dataset_manager.py export <output.json>
    python dataset_manager.py import <input.json>
    python dataset_manager.py stats
"""

import sys, json
from pathlib import Path
from detector import load_dataset, save_dataset


def cmd_list(db: dict):
    if not db:
        print("Dataset is empty.")
        return
    print(f"\n{'Hash':<18}  {'Name':<35}  {'Seen':>6}")
    print("-" * 65)
    for h, v in sorted(db.items(), key=lambda x: x[1]["name"]):
        print(f"{h[:16]:<18}  {v['name']:<35}  {v['seen']:>6}")
    print(f"\nTotal: {len(db)} items")


def cmd_rename(db: dict, hash_prefix: str, new_name: str):
    matches = [k for k in db if k.startswith(hash_prefix)]
    if not matches:
        print(f"No item found with hash prefix: {hash_prefix}")
        return
    if len(matches) > 1:
        print(f"Multiple matches: {matches}")
        return
    key = matches[0]
    old_name = db[key]["name"]
    db[key]["name"] = new_name
    save_dataset(db)
    print(f"✅ Renamed '{old_name}' → '{new_name}'")


def cmd_delete(db: dict, hash_prefix: str):
    matches = [k for k in db if k.startswith(hash_prefix)]
    if not matches:
        print(f"No item found: {hash_prefix}")
        return
    key = matches[0]
    name = db[key]["name"]
    del db[key]
    save_dataset(db)
    print(f"🗑  Deleted: {name} ({key[:16]})")


def cmd_export(db: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"📤 Exported {len(db)} items to {path}")


def cmd_import(db: dict, path: str):
    with open(path, encoding="utf-8") as f:
        incoming = json.load(f)
    merged = {**db, **incoming}
    save_dataset(merged)
    print(f"📥 Imported {len(incoming)} entries. Total: {len(merged)}")


def cmd_stats(db: dict):
    if not db:
        print("Dataset is empty.")
        return
    total     = len(db)
    total_seen = sum(v["seen"] for v in db.values())
    most_seen  = max(db.items(), key=lambda x: x[1]["seen"])
    print(f"\n📊 Dataset Statistics")
    print(f"  Known items : {total}")
    print(f"  Total scans : {total_seen}")
    print(f"  Most seen   : {most_seen[1]['name']} ({most_seen[1]['seen']} times)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    db  = load_dataset()
    cmd = sys.argv[1]

    if cmd == "list":
        cmd_list(db)
    elif cmd == "rename" and len(sys.argv) == 4:
        cmd_rename(db, sys.argv[2], sys.argv[3])
    elif cmd == "delete" and len(sys.argv) == 3:
        cmd_delete(db, sys.argv[2])
    elif cmd == "export" and len(sys.argv) == 3:
        cmd_export(db, sys.argv[2])
    elif cmd == "import" and len(sys.argv) == 3:
        cmd_import(db, sys.argv[2])
    elif cmd == "stats":
        cmd_stats(db)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
