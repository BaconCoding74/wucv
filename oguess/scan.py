"""
Wuthering Waves Inventory Scanner — Interactive CLI
Usage:
    python scan.py <image_path> [--debug] [--auto]
"""

import argparse
import sys
import os
import cv2
import json
from pathlib import Path
from oguess.detector import (
    scan_screenshot, load_dataset, save_dataset,
    register_item, ItemSlot
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False


def print_table(slots: list[ItemSlot]):
    if RICH:
        console = Console()
        table = Table(title="🎮 WuWa Inventory Scan Results", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Name", style="cyan", min_width=20)
        table.add_column("Qty", style="green", justify="right")
        table.add_column("New?", justify="center")
        table.add_column("Hash (short)", style="dim")
        table.add_column("Status", style="yellow")

        for s in slots:
            name   = s.name or "[red]UNKNOWN[/red]"
            is_new = "⭐" if s.is_new else ""
            status = "✅ Known" if s.name else "❓ Need name"
            table.add_row(
                str(s.index),
                name,
                str(s.quantity) if s.quantity else "?",
                is_new,
                s.image_hash[:12],
                status,
            )
        console.print(table)
    else:
        print(f"\n{'#':>3}  {'Name':<25} {'Qty':>6}  {'New':>4}  {'Hash':>14}  Status")
        print("-" * 75)
        for s in slots:
            name   = s.name or "UNKNOWN"
            is_new = "NEW" if s.is_new else ""
            status = "Known" if s.name else "Need name"
            qty    = str(s.quantity) if s.quantity else "?"
            print(f"{s.index:>3}  {name:<25} {qty:>6}  {is_new:>4}  {s.image_hash[:12]:>14}  {status}")
        print()


def interactive_naming(slots: list[ItemSlot], db: dict) -> dict:
    """For each unknown item, show the crop and ask for a name."""
    unknowns = [s for s in slots if s.name is None]
    if not unknowns:
        print("\n✅ All items already identified!")
        return db

    print(f"\n🔍 Found {len(unknowns)} unknown item(s). Let's name them.\n")

    for s in unknowns:
        print(f"\n--- Unknown item #{s.index} ---")
        print(f"  Hash: {s.image_hash[:12]}...")
        print(f"  Qty : {s.quantity}")
        print(f"  Crop: {s.crop_path}")

        # Try to open the crop for viewing
        if s.crop_path and os.path.exists(s.crop_path):
            try:
                cv2.imshow("Unknown Item — Press any key after viewing", cv2.imread(s.crop_path))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                print("  (Could not open image preview — check crop_path manually)")

        if RICH:
            from rich.prompt import Prompt
            name = Prompt.ask("  Enter item name (blank to skip)").strip()
        else:
            name = input("  Enter item name (blank to skip): ").strip()

        if name:
            db = register_item(s.image_hash, name, db)
            s.name = name
            print(f"  ✅ Saved as: {name}")
        else:
            print("  ⏭  Skipped.")

    return db


def export_json(slots: list[ItemSlot], output_path: str):
    data = []
    for s in slots:
        data.append({
            "index":      s.index,
            "name":       s.name,
            "quantity":   s.quantity,
            "is_new":     s.is_new,
            "hash":       s.image_hash,
            "bbox":       list(s.bbox),
            "crop_path":  s.crop_path,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📄 JSON exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WuWa Inventory Scanner")
    parser.add_argument("image", help="Path to the inventory screenshot")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug image with detected slot boxes")
    parser.add_argument("--auto", action="store_true",
                        help="Skip interactive naming (just output results)")
    parser.add_argument("--output", default=None,
                        help="Export results as JSON to this path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    print(f"\n🔎 Scanning: {args.image}")
    slots = scan_screenshot(args.image, debug=args.debug)

    if not slots:
        print("❌ No item slots detected. Try --debug to inspect the grid detection.")
        sys.exit(1)

    print(f"✅ Detected {len(slots)} item slot(s).\n")
    print_table(slots)

    db = load_dataset()

    if not args.auto:
        db = interactive_naming(slots, db)
        # Re-print updated table
        print("\n📋 Updated results:")
        print_table(slots)

    output_path = args.output or args.image.replace(".png", "_scan.json").replace(".jpg", "_scan.json")
    export_json(slots, output_path)

    if args.debug:
        print("\n🐛 Debug slot image saved to: /tmp/wuwa_debug_slots.png")


if __name__ == "__main__":
    main()