#!/usr/bin/env python3
"""Check what role sets are available for Math domain."""
import sys
sys.path.insert(0, '/home/ah872032.ucf/system-aware-mas')
from MAR.InfraMind.inframind_router import _default_role_sets, _load_role_profiles

profiles = _load_role_profiles("Math")
print("=== Math Role Profiles ===")
for name in sorted(profiles.keys()):
    print(f"  {name}")

print(f"\nTotal roles: {len(profiles)}")

role_names = list(profiles.keys())
role_sets = _default_role_sets(role_names)
print(f"\n=== Default Role Sets ({len(role_sets)}) ===")
for i, rs in enumerate(role_sets):
    print(f"  [{i}] {' + '.join(rs)}")
