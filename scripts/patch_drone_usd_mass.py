"""Patch drone USD mass/inertia for Sim 5.1.

Run inside the `isaacsim` conda env (uses `pxr`):

    conda activate isaacsim
    python scripts/patch_drone_usd_mass.py

Sets body mass to 0.5 kg with diagonal inertia (0.003, 0.003, 0.006) and each
prop to 0.025 kg with (1e-6, 1e-6, 2e-6). Props previously had no <inertial>
in URDF, causing Isaac Sim to assign defaults that overshot the drone mass by
~1000x. See HANDOFF_TO_OTHER_REPO.md §2.
"""

import os

from pxr import Gf, Usd, UsdPhysics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USD_PATHS = [
    os.path.join(REPO_ROOT, "assets/5_in_drone/5_in_drone.usd"),
    os.path.join(REPO_ROOT, "assets/5_in_drone/configuration/5_in_drone_physics.usd"),
]


def patch_stage(usd_path: str) -> None:
    print(f"[patch] opening {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"[patch]   SKIP — could not open stage")
        return

    changed = 0
    for prim in stage.TraverseAll():
        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api:
            continue
        name = str(prim.GetPath()).split("/")[-1]
        if name == "body":
            mass_api.GetMassAttr().Set(0.5)
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.003, 0.003, 0.006))
            print(f"[patch]   body         m=0.5 I=(0.003, 0.003, 0.006)")
            changed += 1
        elif name.startswith("prop"):
            mass_api.GetMassAttr().Set(0.025)
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1e-6, 1e-6, 2e-6))
            print(f"[patch]   {name:<12} m=0.025 I=(1e-6, 1e-6, 2e-6)")
            changed += 1

    if changed:
        stage.GetRootLayer().Save()
        print(f"[patch]   SAVED — {changed} prims updated")
    else:
        print(f"[patch]   no MassAPI prims found — nothing to do")


if __name__ == "__main__":
    for path in USD_PATHS:
        if not os.path.isfile(path):
            print(f"[patch] MISSING: {path}")
            continue
        patch_stage(path)
    print("[patch] done")
