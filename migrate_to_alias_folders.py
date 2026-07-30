"""One-time migration: rename client folders from real names to aliases.

Run this once to convert:
  ~/RWM/Current Client Plans/Jones, Bob & Mary/  →  ~/RWM/Current Client Plans/Arctic-Hill/

Also renames plan files inside each folder to generic names (plan_1.json, plan_2.json)
and strips any 'client' field from plan JSONs that contains real names.

Usage:
    python migrate_to_alias_folders.py          # dry run (shows what would happen)
    python migrate_to_alias_folders.py --apply  # actually rename
"""

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.expanduser('~/RWM/pseudonymizer app'))
from main import _load_alias_map, _save_alias_map, CLIENT_DIR


def migrate(apply=False):
    alias_map = _load_alias_map()
    if not alias_map:
        print("No aliases found. Run 'python pseudonymize.py init' first.")
        return

    print(f"{'APPLYING' if apply else 'DRY RUN'} — {len(alias_map)} aliases\n")

    for alias, real_name in sorted(alias_map.items()):
        old_dir = os.path.join(CLIENT_DIR, real_name)
        new_dir = os.path.join(CLIENT_DIR, alias)

        if not os.path.isdir(old_dir):
            if os.path.isdir(new_dir):
                print(f"  {alias}: already migrated")
            else:
                print(f"  {alias}: no folder found (skipping)")
            continue

        print(f"  {real_name}  →  {alias}/")

        # Collect plan files to rename
        plan_files = sorted([
            f for f in os.listdir(old_dir)
            if f.endswith('.json') and not f.endswith('_results.json') and not f.endswith('_plan.json')
        ])

        for i, pf in enumerate(plan_files, 1):
            new_name = f"plan_{i}.json" if len(plan_files) > 1 else "plan_1.json"
            print(f"    {pf}  →  {new_name}")

            if apply:
                # Read plan, strip real name from 'client' field
                plan_path = os.path.join(old_dir, pf)
                try:
                    with open(plan_path) as fh:
                        data = json.load(fh)
                    if 'client' in data:
                        data['client'] = alias
                    if 'client_select' in data:
                        data['client_select'] = alias
                    with open(plan_path, 'w') as fh:
                        json.dump(data, fh, indent=2, default=str)
                except Exception as e:
                    print(f"      WARNING: could not update {pf}: {e}")

                # Rename the file
                os.rename(plan_path, os.path.join(old_dir, new_name))

            # Handle companion files (_plan.json, _results.json, .pdf)
            stem = pf[:-5]  # remove .json
            for suffix in ['_plan.json', '_results.json', '.pdf']:
                companion = stem + suffix
                companion_path = os.path.join(old_dir, companion)
                if os.path.exists(companion_path):
                    new_companion = f"plan_{i}" + suffix
                    print(f"    {companion}  →  {new_companion}")
                    if apply:
                        os.rename(companion_path, os.path.join(old_dir, new_companion))

        # Handle audit.log and backups (keep as-is, just move with folder)

        if apply:
            os.rename(old_dir, new_dir)
            print(f"    Folder renamed.\n")
        else:
            print()

    if not apply:
        print("\nThis was a dry run. Run with --apply to make changes.")


if __name__ == '__main__':
    apply = '--apply' in sys.argv
    migrate(apply=apply)
