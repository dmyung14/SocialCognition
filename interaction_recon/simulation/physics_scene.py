"""Physics MJCF, preserving the retargeted human joint addresses."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.002
    density: float = 600.0
    friction: float = 0.8
    hand_friction: float = 1.0
    contact_timeconst: float = 0.01
    weld_timeconst: float = 0.04
    finger_kp: float = 8.0
    finger_kv: float = 0.35
    finger_force: float = 1.0
    wrist_kp: float = 20.0
    wrist_kv: float = 1.0
    wrist_force: float = 3.0
    hand_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    timing_offset_s: float = 0.0
    preroll_s: float = 0.4
    startup_s: float = 0.4
    initial_confidence: float = 0.05
    width: int = 640
    height: int = 480

    def __post_init__(self):
        if self.timestep not in (0.001, 0.002, 0.004):
            raise ValueError("Physics timestep must be 0.001, 0.002, or coarse 0.004")
        for name, low, high in (
            ("density", 300, 1200),
            ("friction", 0.3, 1.2),
            ("hand_friction", 0.5, 1.5),
            ("contact_timeconst", 0.005, 0.03),
            ("weld_timeconst", 0.02, 0.2),
            ("timing_offset_s", -0.15, 0.15),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low},{high}]")
        offset = np.asarray(self.hand_offset_m, float)
        if (
            offset.shape != (3,) or not np.isfinite(offset).all()
            or np.linalg.norm(offset) > 0.03 + 1e-12
        ):
            raise ValueError("hand_offset_m must be a finite translation of length <=3cm")
        for name in (
            "finger_kp", "finger_kv", "finger_force",
            "wrist_kp", "wrist_kv", "wrist_force",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.2 <= self.preroll_s <= 2 or not 0.1 <= self.startup_s <= 2:
            raise ValueError("Invalid pre-roll/startup duration")
        if not 0 <= self.initial_confidence <= 1:
            raise ValueError("Invalid initialization confidence")
        if min(self.width, self.height) < 64 or self.width % 2 or self.height % 2:
            raise ValueError("Render dimensions must be even and at least 64")


def builder_hand_geom(model: mujoco.MjModel, geom: int) -> bool:
    body_name = model.body(int(model.geom_bodyid[geom])).name or ""
    geom_name = model.geom(geom).name or ""
    return bool(
        body_name.startswith("builder_")
        and (
            body_name.endswith("_palm")
            or any(f"_{finger}_" in body_name for finger in (
                "thumb", "index", "middle", "ring", "little"
            ))
            or "fingertip" in geom_name
            or "hand" in geom_name
        )
    )


def physics_variant(xml: str, config: PhysicsConfig = PhysicsConfig()) -> str:
    root = ET.fromstring(xml)
    root.set("model", "interaction_contact_physics")
    option = root.find("option")
    option.set("timestep", str(config.timestep))
    option.set("integrator", "implicitfast")
    option.set("iterations", "50")
    option.set("cone", "elliptic")
    world = root.find("worldbody")
    equality = ET.SubElement(root, "equality")
    actuators = ET.SubElement(root, "actuator")
    solref = f"{config.contact_timeconst} 1"

    # MuJoCo normally takes the maximum sliding friction of two geoms.
    # Priorities make table and hand friction independently identifiable without
    # explicit contact pairs (which the no-block-constraint audit disallows).
    table = world.find("geom[@name='table']")
    table.set("priority", "2")
    table.set("friction", f"{config.friction} 0.02 0.002")
    table.set("solref", solref)
    table.set("condim", "4")

    for body in world.iter("body"):
        name = body.get("name", "")
        if name.startswith("guider_"):
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        if name.startswith("block_"):
            for geom in body.findall("geom"):
                geom.set("density", str(config.density))
                geom.set("priority", "1")
                geom.set("friction", f"{config.friction} 0.02 0.002")
                geom.set("condim", "4")
                geom.set("solref", solref)
        if name.startswith("builder_"):
            for geom in body.findall("geom"):
                geom.set("priority", "3")
                geom.set("condim", "4")
                geom.set("friction", f"{config.hand_friction} 0.02 0.002")
                geom.set("solref", solref)
            for joint in body.findall("joint"):
                joint_name = joint.get("name")
                wrist = "_wrist_" in joint_name
                joint.set("damping", "0.15" if wrist else "0.04")
                joint.set("armature", "0.0005")
                force = config.wrist_force if wrist else config.finger_force
                ET.SubElement(
                    actuators, "position", name=f"drive_{joint_name}",
                    joint=joint_name,
                    kp=str(config.wrist_kp if wrist else config.finger_kp),
                    kv=str(config.wrist_kv if wrist else config.finger_kv),
                    ctrllimited="true", ctrlrange=joint.get("range"),
                    forcelimited="true", forcerange=f"{-force} {force}",
                )
    for side in ("left", "right"):
        body_name = f"builder_{side}_forearm"
        body = world.find(f"body[@name='{body_name}']")
        target_name = f"target_builder_{side}"
        ET.SubElement(
            world, "body", name=target_name, mocap="true",
            pos=body.get("pos", "0 0 0"), quat=body.get("quat", "1 0 0 0"),
        )
        ET.SubElement(
            equality, "weld", name=f"drive_builder_{side}_root",
            body1=target_name, body2=body_name,
            relpose="0 0 0 1 0 0 0",
            solref=f"{config.weld_timeconst} 1",
            solimp="0.9 0.95 0.001",
        )
    ET.indent(root)
    result = ET.tostring(root, encoding="unicode")
    audit_physics_scene(result)
    return result


def audit_physics_scene(xml: str) -> mujoco.MjModel:
    """Fail closed: only builder scalar actuators and builder root welds."""
    root = ET.fromstring(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    parents = {child: parent for parent in root.iter() for child in parent}
    world = root.find("worldbody")
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if not name.startswith("block_"):
            continue
        if parents[body] is not world or body.get("mocap", "false") != "false":
            raise ValueError("Block must be a non-mocap world child")
        if len(body.findall("freejoint")) != 1 or body.findall("joint") or body.findall("body"):
            raise ValueError("Block must have exactly one freejoint and no child body")
        bid = model.body(name).id
        if model.body_mocapid[bid] >= 0:
            raise ValueError("Mocap block forbidden")
        if model.body_mass[bid] <= 0:
            raise ValueError("Block mass must be positive")
        for geom in body.findall("geom"):
            gid = model.geom(geom.get("name")).id
            if not model.geom_contype[gid] or not model.geom_conaffinity[gid]:
                raise ValueError("Block contacts must be enabled")
    for container in ("actuator", "equality", "tendon", "contact"):
        section = root.find(container)
        if section is None:
            continue
        for element in section.iter():
            for attribute, value in element.attrib.items():
                if attribute != "name" and any(
                    token.startswith("block_") for token in value.split()
                ):
                    raise ValueError(f"Forbidden {container} reference to a block")
    tendons = root.find("tendon")
    if tendons is not None and len(tendons):
        raise ValueError("Physics baseline does not authorize tendons")
    for actuator in root.findall("./actuator/*"):
        joint = actuator.get("joint", "")
        if actuator.tag != "position" or not joint.startswith("builder_"):
            raise ValueError("Only builder joint position actuators are authorized")
        jid = model.joint(joint).id
        if model.jnt_type[jid] not in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
        ):
            raise ValueError("Only scalar builder joints may be actuated")
    for equality in root.findall("./equality/*"):
        if (
            equality.tag != "weld"
            or equality.get("body1") not in ("target_builder_left", "target_builder_right")
            or equality.get("body2") not in ("builder_left_forearm", "builder_right_forearm")
        ):
            raise ValueError("Only builder-root compliant welds are authorized")
    if model.nu and (
        not np.all(model.actuator_ctrllimited)
        or not np.all(model.actuator_forcelimited)
        or not np.isfinite(model.actuator_forcerange).all()
    ):
        raise ValueError("Controllers require finite control and force limits")
    return model
