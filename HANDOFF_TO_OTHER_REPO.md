# Handoff: Fixes to Apply in Other Repo

## Fixes to apply

### 1. ContactSensor phantom in Sim 5.1 (most important)

Sim 5.1's `ContactSensor.net_forces_w` reports a constant phantom on the drone body even at rest (76 N with broken masses, 1869 N with correct masses). Fix in scene cfg — narrow sensor + add buffer params:

```python
collision_sensor: ContactSensorCfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body",   # body only, drop props (stale data per NVIDIA PhysX docs)
    history_length=3,
    update_period=0.0,
    force_threshold=10.0,
    debug_vis=False,
)
```

Termination:

```python
collision = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 10.0},
)
```

### 2. Asset USD mass/inertia (props had no `<inertial>` → 1000× over-mass)

URDF source patch — add to all 4 prop links:

```xml
<inertial>
  <origin rpy="0 0 0" xyz="0 0 0"/>
  <mass value="0.025"/>
  <inertia ixx="1e-6" ixy="0.0" ixz="0.0" iyy="1e-6" iyz="0.0" izz="2e-6"/>
</inertial>
```

USD patch script (run in conda `isaacsim` env):

```python
from pxr import Usd, UsdPhysics, Gf
for usd in ['assets/5_in_drone/5_in_drone.usd', 'assets/5_in_drone/configuration/5_in_drone_physics.usd']:
    stage = Usd.Stage.Open(usd)
    for prim in stage.TraverseAll():
        m = UsdPhysics.MassAPI(prim)
        if not m: continue
        n = str(prim.GetPath()).split('/')[-1]
        if n == 'body':
            m.GetMassAttr().Set(0.5)
            m.GetCenterOfMassAttr().Set(Gf.Vec3f(0,0,0))
            m.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.003, 0.003, 0.006))
        elif n.startswith('prop'):
            m.GetMassAttr().Set(0.025)
            m.GetCenterOfMassAttr().Set(Gf.Vec3f(0,0,0))
            m.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1e-6, 1e-6, 2e-6))
    stage.GetRootLayer().Save()
```

### 3. Zero initial motor joint_vel in `assets/five_in_drone.py`

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},          # was {m1_joint:200, m2_joint:-200, ...}
),
```

### 4. PLAY respawn fix — train/play parity

Training cfgs (`DroneRacerEnvCfg.__post_init__`, `_NoCam.__post_init__`):

```python
self.events.reset_base = None
self.commands.target.randomise_start = True
```

PLAY cfgs — leave both at defaults (no override). PLAY then uses fixed `reset_base` corner spawn + `randomise_start=None` → predictable respawn at gate 0. No respawn loop.

### 5. Spawn config — match upstream PPO

`CommandsCfg.target`:

```python
spawn_lerp_alpha=0.0,
spawn_forward_offset=1.0,
spawn_forward_velocity=0.0,
```

Add field to `GateTargetingCommandCfg` (likely missing in dreamer repo too):

```python
spawn_forward_offset: float = 0.0
```

Thread through `reset_after_prev_gate(..., forward_offset=self.cfg.spawn_forward_offset)` in `commands.py:_resample_command`.

### 6. Upstream-tuned reward weights (Dreamer values break skrl PPO)

`RewardsCfg`:

```python
terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.0001)
progress = RewTerm(func=mdp.progress, weight=20.0,
                   params={"command_name": "target", "asymmetric": False})
gate_passed = RewTerm(func=mdp.gate_passed, weight=400.0,
                      params={"command_name": "target", "penalize_miss": True})
lookat_next = RewTerm(func=mdp.lookat_next_gate, weight=0.1,
                      params={"command_name": "target", "std": 0.5})
```

`TerminationsCfg`:

```python
flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 20.0})
```

`EventCfg.push_robot` — enable (was disabled):

```python
push_robot = EventTerm(func=mdp.apply_external_force_torque, mode="interval",
                       interval_range_s=(0.0, 0.2),
                       params={"force_range": (-0.1, 0.1), "torque_range": (-0.05, 0.05)})
```

### 7. Reward function signature changes

- `mdp.progress`: add `asymmetric: bool = False` arg. When `True` clamp `min=0`, else signed.
- `mdp.gate_passed`: add `penalize_miss: bool = True` arg. When `True` return `passed - missed`, else just `passed`.

### 8. NoCam_PLAY camera disable

```python
def __post_init__(self):
    ...
    self.scene.tiled_camera = None   # must run without --enable_cameras
```

### 9. Z range typo

`EventCfg.reset_base.pose_range["z"]`: `(1.5, 0.5)` → `(0.5, 1.5)`. Functionally same (`sample_uniform` handles reversed) but readable.

### 10. CTBR action variant (if using CTBR)

`mdp.actions.py` — add `CTBRAction` + `CTBRActionCfg` (PD rate controller, applies thrust + torques via `permanent_wrench_composer`). Default `gains_path="tasks/drone_racer/configs/ctbr_gains.yaml"`.

`gains_path` YAML:

```yaml
roll:  {kp: 0.964, kd: 0.0}
pitch: {kp: 0.72,  kd: 0.0}
yaw:   {kp: 1.5036, kd: 0.0}
max_roll_rate: 10.0
max_pitch_rate: 10.0
max_yaw_rate: 2.0
max_thrust: 23.8239
hover_thrust: 5.9606
```

## Order to apply

1. URDF + USD mass fix (asset cleanup)
2. Reward function param additions (`asymmetric`, `penalize_miss`)
3. `GateTargetingCommandCfg.spawn_forward_offset` field + wiring
4. ContactSensor cfg narrowed + `history_length`/`update_period`
5. Env cfg: rewards, events, terminations, commands, train/play `__post_init__`
6. Retrain from scratch — old checkpoints poisoned by broken collision termination

## Reference commits in this repo

- `253545e` — asset mass/inertia
- `618fcb6` — init joint_vel zero
- `e288c81` + `15af300` — sensor cfg + threshold
- `af470ad` — rewards
- `93da8e7` — spawn_forward_offset
- `85a417e` — PLAY respawn fix
