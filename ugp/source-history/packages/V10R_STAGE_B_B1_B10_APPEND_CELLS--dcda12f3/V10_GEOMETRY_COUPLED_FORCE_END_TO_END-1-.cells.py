# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
# ======================================================================
# FORCE-5 NEW-RUNTIME BOOTSTRAP
#
# RUN THIS ONCE IN A FRESH COLAB RUNTIME
# BEFORE RUNNING THE LARGE FORCE-5A CELL.
#
# It does NOT retrain V7/V8/V9.
# It restores the persistent experiment state needed by FORCE-5A.
# ======================================================================

import os
import sys
import gc
import json
import time
import shutil
import random
import subprocess
import warnings
import zipfile

from pathlib import Path

import numpy as np
import pandas as pd

import torch

from IPython.display import display


print("=" * 112)
print("FORCE-5 — FRESH COLAB RUNTIME BOOTSTRAP")
print("=" * 112)


# ======================================================================
# 1. GPU CHECK
# ======================================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "GPU is not enabled.\n"
        "Colab -> Runtime -> Change runtime type -> T4 GPU, "
        "then rerun this cell."
    )


DEVICE = torch.device(
    "cuda"
)


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "Torch:",
    torch.__version__
)


# ======================================================================
# 2. MOUNT GOOGLE DRIVE
# ======================================================================

from google.colab import drive


if not Path(
    "/content/drive/MyDrive"
).exists():

    drive.mount(
        "/content/drive"
    )

else:

    print(
        "✅ Google Drive already mounted"
    )


MYDRIVE = Path(
    "/content/drive/MyDrive"
)


if not MYDRIVE.exists():

    raise RuntimeError(
        "Google Drive mount failed."
    )


# ======================================================================
# 3. PROJECT PATHS
# ======================================================================

PERSIST_ROOT = (
    MYDRIVE /
    "UGP_V7_FULL4B"
)

CACHE_ROOT = (
    PERSIST_ROOT /
    "cache"
)

OUT_ROOT = (
    PERSIST_ROOT /
    "results"
)

V8_ROOT = (
    PERSIST_ROOT /
    "V8_contactmap_standalone"
)

V9_ROOT = (
    PERSIST_ROOT /
    "V9_contact_belief_memory"
)

V9_A_ROOT = (
    V9_ROOT /
    "A_belief_memory"
)

V9_B_ROOT = (
    V9_ROOT /
    "B_contact_events"
)

V9_C_ROOT = (
    V9_ROOT /
    "C_residual_dynamics"
)


for p in [
    PERSIST_ROOT,
    CACHE_ROOT,
    OUT_ROOT,
    V8_ROOT,
    V9_ROOT,
    V9_A_ROOT,
    V9_B_ROOT,
    V9_C_ROOT,
]:

    p.mkdir(
        parents=True,
        exist_ok=True
    )


print()
print(
    "PERSIST_ROOT:",
    PERSIST_ROOT
)

print(
    "CACHE_ROOT  :",
    CACHE_ROOT
)

print(
    "V9_ROOT     :",
    V9_ROOT
)


# ======================================================================
# 4. EXACT EXPERIMENT CONFIG
# ======================================================================

EXPERIMENTS = [

    dict(
        terrain_type="forest",
        environment="ForestEnv",
        run="P2000",
        ttests=[
            28.0,
            45.0,
            75.0,
            90.0,
        ],
    ),

    dict(
        terrain_type="swamp_marsh",
        environment="GreatMarsh",
        run="P2023",
        ttests=[
            47.0,
            61.0,
            73.0,
            87.0,
        ],
    ),

    dict(
        terrain_type="rough_uneven",
        environment="SeasonalForestWinter",
        run="P2000",
        ttests=[
            29.0,
            46.0,
            64.0,
            92.0,
        ],
    ),

    dict(
        terrain_type="wet_uneven_proxy",
        environment="OldTownFall",
        run="P2000",
        ttests=[
            32.0,
            84.0,
            100.0,
            115.0,
        ],
    ),
]


display(
    pd.DataFrame(
        EXPERIMENTS
    )
)


# ======================================================================
# 5. EXPECTED CACHE FILES
# ======================================================================

physics_files = []

manifest_files = []

visual_files = []


for e in EXPERIMENTS:

    env = e[
        "environment"
    ]

    run = e[
        "run"
    ]

    physics_files.append(
        CACHE_ROOT /
        f"{env}_{run}_physics.npz"
    )

    manifest_files.append(
        CACHE_ROOT /
        f"{env}_{run}_frame_manifest.npz"
    )

    visual_files.extend(
        [
            CACHE_ROOT /
            f"{env}_{run}_v7_dinov3.npy",

            CACHE_ROOT /
            f"{env}_{run}_v7_convnext.npy",

            CACHE_ROOT /
            f"{env}_{run}_v7_utonia.npy",

            CACHE_ROOT /
            f"{env}_{run}_v7_waft.npy",
        ]
    )


# ======================================================================
# 6. RECOVER MISSING CACHE FILES FROM ANY DUPLICATE LOCATION IN DRIVE
#
# This handles the case where the previous runtime had copied/extracted
# the files elsewhere in MyDrive.
# ======================================================================

def recover_file_from_drive(
    target
):

    if target.exists():

        return True

    print(
        "Searching Drive for:",
        target.name
    )

    matches = [

        p for p in MYDRIVE.rglob(
            target.name
        )

        if (
            p.is_file()
            and
            p.resolve()
            !=
            target.resolve()
        )
    ]

    if matches:

        src = matches[
            0
        ]

        print(
            "  found:",
            src
        )

        shutil.copy2(
            src,
            target
        )

        print(
            "  restored ->",
            target
        )

        return True

    return False


# Physics files are mandatory for FORCE-5A.
for p in physics_files:

    recover_file_from_drive(
        p
    )


# ======================================================================
# 7. FALLBACK: EXTRACT MISSING PHYSICS CACHES FROM ZIP BACKUP
#
# Only runs if a mandatory physics cache is STILL absent.
# Does not blindly extract enormous archives.
# ======================================================================

missing_physics = [

    p for p in physics_files
    if not p.exists()
]


if missing_physics:

    print()
    print(
        "Some physics caches are still missing."
    )

    print(
        "Searching for V7/V8 result ZIP backups..."
    )

    zip_candidates = [

        p for p in MYDRIVE.rglob(
            "*.zip"
        )

        if (
            "V7" in p.name.upper()
            or
            "V8" in p.name.upper()
            or
            "CACHE" in p.name.upper()
        )
    ]

    wanted = {
        p.name: p
        for p in missing_physics
    }

    for zp in zip_candidates:

        if not wanted:

            break

        try:

            with zipfile.ZipFile(
                zp,
                "r"
            ) as zf:

                names = zf.namelist()

                for member in names:

                    basename = Path(
                        member
                    ).name

                    if basename not in wanted:

                        continue

                    target = wanted[
                        basename
                    ]

                    print(
                        "Extracting",
                        basename,
                        "from",
                        zp.name
                    )

                    with zf.open(
                        member
                    ) as src:

                        with open(
                            target,
                            "wb"
                        ) as dst:

                            shutil.copyfileobj(
                                src,
                                dst
                            )

                    del wanted[
                        basename
                    ]

        except Exception as exc:

            print(
                "Skipping unreadable ZIP:",
                zp,
                repr(exc)
            )


missing_physics = [

    p for p in physics_files
    if not p.exists()
]


if missing_physics:

    raise FileNotFoundError(

        "MANDATORY physics caches are missing:\n"
        +
        "\n".join(
            str(p)
            for p in missing_physics
        )
    )


print()
print(
    "✅ all 4 mandatory physics caches found"
)


# ======================================================================
# 8. OPTIONAL VISUAL/MANIFEST RECOVERY
#
# FORCE-5A does NOT require these, but we recover their normal locations
# now so future FORCE-5B / foothold cells can reuse this runtime.
# ======================================================================

for p in (
    manifest_files
    +
    visual_files
):

    if not p.exists():

        recover_file_from_drive(
            p
        )


# ======================================================================
# 9. LOAD NPZ HELPER
# ======================================================================

def load_npz_dict(
    path
):

    z = np.load(
        path,
        allow_pickle=False
    )

    return {
        k: z[k]
        for k in z.files
    }


# ======================================================================
# 10. REBUILD run_records
#
# Physics arrays are loaded normally because FORCE-5A iterates them
# heavily.
#
# Visual caches are attached with mmap_mode='r' if available:
# they do NOT need to sit fully in RAM.
# ======================================================================

run_records = []


for e in EXPERIMENTS:

    env = e[
        "environment"
    ]

    run = e[
        "run"
    ]

    pphysics = (
        CACHE_ROOT /
        f"{env}_{run}_physics.npz"
    )

    data = load_npz_dict(
        pphysics
    )

    rec = dict(
        **e,

        data=data,
    )

    # --------------------------------------------------------------
    # Optional manifest.
    # --------------------------------------------------------------

    pmanifest = (
        CACHE_ROOT /
        f"{env}_{run}_frame_manifest.npz"
    )

    if pmanifest.exists():

        rec[
            "manifest"
        ] = load_npz_dict(
            pmanifest
        )


    # --------------------------------------------------------------
    # Optional visual streams as memory maps.
    #
    # This is essentially free in RAM until data is accessed.
    # --------------------------------------------------------------

    visual_paths = {

        "DINO":
        CACHE_ROOT /
        f"{env}_{run}_v7_dinov3.npy",

        "CONV":
        CACHE_ROOT /
        f"{env}_{run}_v7_convnext.npy",

        "GEOM":
        CACHE_ROOT /
        f"{env}_{run}_v7_utonia.npy",

        "FLOW":
        CACHE_ROOT /
        f"{env}_{run}_v7_waft.npy",
    }


    visual = {}

    for name, path in visual_paths.items():

        if path.exists():

            visual[
                name
            ] = np.load(
                path,
                mmap_mode="r"
            )


    if visual:

        # If manifest exists, add the manifest fields the same way V8 did.
        if "manifest" in rec:

            visual.update(
                rec[
                    "manifest"
                ]
            )

        rec[
            "visual"
        ] = visual


    run_records.append(
        rec
    )


print()
print(
    "✅ run_records rebuilt:",
    len(run_records)
)


# ======================================================================
# 11. STRICT PHYSICS DATA CONTRACT
# ======================================================================

expected_shapes = {

    ("ForestEnv", "P2000"):
    12514,

    ("GreatMarsh", "P2023"):
    12927,

    ("SeasonalForestWinter", "P2000"):
    12908,

    ("OldTownFall", "P2000"):
    12516,
}


contract_rows = []


for rec in run_records:

    d = rec[
        "data"
    ]

    env = rec[
        "environment"
    ]

    run = rec[
        "run"
    ]

    required_keys = [
        "time",
        "P",
        "B",
        "U",
        "L_LEG",
        "E",
        "contact",
    ]

    missing = [

        k for k in required_keys
        if k not in d
    ]

    if missing:

        raise RuntimeError(
            f"{env}/{run} missing arrays: {missing}"
        )


    N = len(
        d[
            "time"
        ]
    )


    if (
        env,
        run
    ) in expected_shapes:

        expected_N = expected_shapes[
            (
                env,
                run
            )
        ]

        # Allow tiny historical cache differences, but not a corrupt cache.
        if abs(
            N - expected_N
        ) > 5:

            raise RuntimeError(

                f"{env}/{run}: N={N}, "
                f"expected about {expected_N}"
            )


    if d["P"].shape != (
        N,
        4,
        12
    ):

        raise RuntimeError(
            f'{env}/{run}: bad P shape {d["P"].shape}'
        )


    if d["B"].shape != (
        N,
        12
    ):

        raise RuntimeError(
            f'{env}/{run}: bad B shape {d["B"].shape}'
        )


    if d["U"].shape != (
        N,
        6
    ):

        raise RuntimeError(
            f'{env}/{run}: bad U shape {d["U"].shape}'
        )


    if d["L_LEG"].shape != (
        N,
        4,
        3
    ):

        raise RuntimeError(
            f'{env}/{run}: bad L_LEG shape {d["L_LEG"].shape}'
        )


    if d["contact"].shape != (
        N,
        4
    ):

        raise RuntimeError(
            f'{env}/{run}: bad contact shape {d["contact"].shape}'
        )


    for k in [
        "time",
        "P",
        "B",
        "U",
        "L_LEG",
        "contact",
    ]:

        if not np.isfinite(
            d[k]
        ).all():

            raise RuntimeError(
                f"{env}/{run}: nonfinite values in {k}"
            )


    dt = float(
        np.median(
            np.diff(
                d[
                    "time"
                ]
            )
        )
    )


    contract_rows.append(
        dict(
            terrain_type=rec[
                "terrain_type"
            ],

            environment=env,

            run=run,

            samples=N,

            dt_s=dt,

            duration_s=float(
                d[
                    "time"
                ][
                    -1
                ]
                -
                d[
                    "time"
                ][
                    0
                ]
            ),

            loaded_contact_fraction=float(
                np.mean(
                    d[
                        "contact"
                    ]
                )
            ),

            visual_loaded=(
                "visual"
                in rec
            ),
        )
    )


contract_df = pd.DataFrame(
    contract_rows
)


display(
    contract_df
)


print(
    "✅ physics data contract verified"
)


# ======================================================================
# 12. RESTORE OFFICIAL ANYMAL-D URDF
#
# New Colab runtime lost /content files, so clone only this tiny
# description repository. We are NOT reinstalling Utonia/WAFT/etc.
# ======================================================================

import xml.etree.ElementTree as ET


EXT = Path(
    "/content/force5_external"
)

EXT.mkdir(
    parents=True,
    exist_ok=True
)


ANYMAL_D_DESC = (
    EXT /
    "anymal_d_simple_description"
)


if not ANYMAL_D_DESC.exists():

    print()
    print(
        "Cloning official ANYmal-D simple description..."
    )

    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ANYbotics/anymal_d_simple_description.git",
            str(
                ANYMAL_D_DESC
            ),
        ]
    )


ANYMAL_D_URDF = (
    ANYMAL_D_DESC /
    "urdf" /
    "anymal.urdf"
)


if not ANYMAL_D_URDF.exists():

    raise RuntimeError(
        "ANYmal-D URDF clone incomplete."
    )


print(
    "✅ ANYmal-D URDF:",
    ANYMAL_D_URDF
)


# ======================================================================
# 13. EXACT ORIGINAL VERIFIED ANYMAL-D FK
# ======================================================================

def _rpy_R(
    rpy
):

    r, p, y = [
        float(v)
        for v in rpy
    ]

    cr, sr = (
        np.cos(r),
        np.sin(r)
    )

    cp, sp = (
        np.cos(p),
        np.sin(p)
    )

    cy, sy = (
        np.cos(y),
        np.sin(y)
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr],
        ],
        np.float64
    )

    Ry = np.array(
        [
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp],
        ],
        np.float64
    )

    Rz = np.array(
        [
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1],
        ],
        np.float64
    )

    return (
        Rz @ Ry @ Rx
    )


def _axis_R(
    axis,
    angle
):

    a = np.asarray(
        axis,
        np.float64
    )

    a = (
        a /
        (
            np.linalg.norm(a)
            +
            1e-12
        )
    )

    x, y, z = a

    c = np.cos(
        angle
    )

    ss = np.sin(
        angle
    )

    C = (
        1 - c
    )

    return np.array(
        [
            [
                c + x*x*C,
                x*y*C - z*ss,
                x*z*C + y*ss,
            ],

            [
                y*x*C + z*ss,
                c + y*y*C,
                y*z*C - x*ss,
            ],

            [
                z*x*C - y*ss,
                z*y*C + x*ss,
                c + z*z*C,
            ],
        ],
        np.float64
    )


class StrictURDFFK:

    def __init__(
        self,
        urdf_path,
        leg_prefixes=(
            "LF",
            "RF",
            "LH",
            "RH",
        ),
    ):

        self.urdf_path = Path(
            urdf_path
        )

        root = ET.parse(
            self.urdf_path
        ).getroot()

        self.links = {
            x.attrib["name"]
            for x in root.findall(
                "link"
            )
        }

        self.joints = {}

        self.child_to_joint = {}

        all_children = set()


        for je in root.findall(
            "joint"
        ):

            name = je.attrib[
                "name"
            ]

            typ = je.attrib.get(
                "type",
                "fixed"
            )

            parent = (
                je.find(
                    "parent"
                )
                .attrib[
                    "link"
                ]
            )

            child = (
                je.find(
                    "child"
                )
                .attrib[
                    "link"
                ]
            )

            org = je.find(
                "origin"
            )

            xyz = np.zeros(
                3
            )

            rpy = np.zeros(
                3
            )

            if org is not None:

                if "xyz" in org.attrib:

                    xyz = np.fromstring(
                        org.attrib[
                            "xyz"
                        ],
                        sep=" ",
                        dtype=np.float64
                    )

                if "rpy" in org.attrib:

                    rpy = np.fromstring(
                        org.attrib[
                            "rpy"
                        ],
                        sep=" ",
                        dtype=np.float64
                    )


            axis_e = je.find(
                "axis"
            )

            axis = np.array(
                [
                    1.0,
                    0.0,
                    0.0
                ]
            )

            if (
                axis_e is not None
                and
                "xyz" in axis_e.attrib
            ):

                axis = np.fromstring(
                    axis_e.attrib[
                        "xyz"
                    ],
                    sep=" ",
                    dtype=np.float64
                )


            d = dict(
                name=name,
                type=typ,
                parent=parent,
                child=child,
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )

            self.joints[
                name
            ] = d

            self.child_to_joint[
                child
            ] = d

            all_children.add(
                child
            )


        roots = list(
            self.links
            -
            all_children
        )

        if len(
            roots
        ) != 1:

            raise RuntimeError(
                "ANYmal-D URDF must have one root link; "
                f"got {roots}"
            )


        self.root_link = (
            roots[
                0
            ]
        )

        self.leg_prefixes = tuple(
            leg_prefixes
        )

        self.foot_links = []

        self.chains = []


        for pref in self.leg_prefixes:

            cands = [

                l for l in self.links

                if (
                    l.upper().startswith(
                        pref + "_"
                    )
                    and
                    "FOOT"
                    in
                    l.upper()
                )
            ]

            valid = []


            for link in cands:

                try:

                    chain = (
                        self._chain_to_root(
                            link
                        )
                    )

                except Exception:

                    continue


                names = [
                    j[
                        "name"
                    ]
                    for j in chain
                ]

                required = [
                    f"{pref}_HAA",
                    f"{pref}_HFE",
                    f"{pref}_KFE",
                ]


                if all(
                    q in names
                    for q in required
                ):

                    valid.append(
                        (
                            len(
                                chain
                            ),
                            link,
                            chain,
                        )
                    )


            if not valid:

                raise RuntimeError(
                    "Could not identify verified "
                    f"{pref} foot link."
                )


            _, link, chain = max(
                valid,
                key=lambda x: x[
                    0
                ]
            )

            self.foot_links.append(
                link
            )

            self.chains.append(
                chain
            )


        expected = [

            f"{p}_{j}"

            for p in self.leg_prefixes

            for j in (
                "HAA",
                "HFE",
                "KFE",
            )
        ]


        missing = [

            j for j in expected

            if j not in self.joints
        ]


        if missing:

            raise RuntimeError(
                "Official ANYmal-D URDF missing joints: "
                +
                repr(
                    missing
                )
            )


        print(
            "ANYmal-D FK root:",
            self.root_link
        )

        print(
            "ANYmal-D FK feet:",
            dict(
                zip(
                    self.leg_prefixes,
                    self.foot_links
                )
            )
        )


    def _chain_to_root(
        self,
        link
    ):

        chain = []

        cur = link

        guard = 0


        while (
            cur
            !=
            self.root_link
        ):

            if (
                cur
                not in
                self.child_to_joint
            ):

                raise RuntimeError(
                    f"No joint connects {cur} "
                    f"to root {self.root_link}"
                )


            j = self.child_to_joint[
                cur
            ]

            chain.append(
                j
            )

            cur = j[
                "parent"
            ]

            guard += 1


            if guard > 100:

                raise RuntimeError(
                    "URDF cycle detected"
                )


        return list(
            reversed(
                chain
            )
        )


    @staticmethod
    def _T(
        R=None,
        t=None
    ):

        T = np.eye(
            4,
            dtype=np.float64
        )


        if R is not None:

            T[
                :3,
                :3
            ] = R


        if t is not None:

            T[
                :3,
                3
            ] = t


        return T


    def foot_positions(
        self,
        q_4x3
    ):

        q = np.asarray(
            q_4x3,
            np.float64
        )


        if q.shape != (
            4,
            3
        ):

            raise ValueError(
                "Expected q shape (4,3), "
                f"got {q.shape}"
            )


        qmap = {

            f"{p}_{j}":
            float(
                q[
                    i,
                    k
                ]
            )

            for i, p
            in enumerate(
                self.leg_prefixes
            )

            for k, j
            in enumerate(
                (
                    "HAA",
                    "HFE",
                    "KFE",
                )
            )
        }


        out = []


        for chain in self.chains:

            T = np.eye(
                4,
                dtype=np.float64
            )


            for j in chain:

                T = (
                    T
                    @
                    self._T(
                        _rpy_R(
                            j[
                                "rpy"
                            ]
                        ),
                        j[
                            "xyz"
                        ]
                    )
                )


                if j[
                    "type"
                ] in (
                    "revolute",
                    "continuous",
                ):

                    T = (
                        T
                        @
                        self._T(
                            _axis_R(
                                j[
                                    "axis"
                                ],
                                qmap.get(
                                    j[
                                        "name"
                                    ],
                                    0.0
                                )
                            ),
                            None
                        )
                    )


                elif (
                    j[
                        "type"
                    ]
                    ==
                    "prismatic"
                ):

                    T = (
                        T
                        @
                        self._T(
                            None,
                            j[
                                "axis"
                            ]
                            *
                            qmap.get(
                                j[
                                    "name"
                                ],
                                0.0
                            )
                        )
                    )


                elif (
                    j[
                        "type"
                    ]
                    !=
                    "fixed"
                ):

                    raise RuntimeError(
                        "Unsupported URDF joint type: "
                        +
                        repr(
                            j[
                                "type"
                            ]
                        )
                    )


            out.append(
                T[
                    :3,
                    3
                ]
            )


        out = np.asarray(
            out,
            np.float32
        )


        if (
            not np.isfinite(
                out
            ).all()
            or
            np.max(
                np.linalg.norm(
                    out,
                    axis=1
                )
            )
            >
            2.0
        ):

            raise RuntimeError(
                "ANYmal-D FK produced "
                "implausible foot positions: "
                +
                repr(
                    out
                )
            )


        return out


ANYMAL_D_FK = StrictURDFFK(
    ANYMAL_D_URDF
)


# ======================================================================
# 14. FK SANITY CHECK AGAINST ALL FOUR RUNS
# ======================================================================

fk_rows = []


for rec in run_records:

    q = np.asarray(
        rec[
            "data"
        ][
            "P"
        ][
            0,
            :,
            0:3
        ],
        np.float32
    )

    feet = (
        ANYMAL_D_FK
        .foot_positions(
            q
        )
    )


    if feet.shape != (
        4,
        3
    ):

        raise RuntimeError(
            "FK returned wrong shape."
        )


    fk_rows.append(
        dict(
            terrain_type=rec[
                "terrain_type"
            ],

            FL=feet[
                0
            ].tolist(),

            FR=feet[
                1
            ].tolist(),

            RL=feet[
                2
            ].tolist(),

            RR=feet[
                3
            ].tolist(),
        )
    )


display(
    pd.DataFrame(
        fk_rows
    )
)


print(
    "✅ ANYmal-D FK verified"
)


# ======================================================================
# 15. CHECK COMMAND TORQUE INFORMATION
# ======================================================================

command_rows = []


for rec in run_records:

    d = rec[
        "data"
    ]

    L = np.asarray(
        d[
            "L_LEG"
        ],
        np.float32
    )

    tau = np.asarray(
        d[
            "P"
        ][
            :,
            :,
            6:9
        ],
        np.float32
    )


    command_rows.append(
        dict(
            terrain_type=rec[
                "terrain_type"
            ],

            commanded_tau_abs_mean=float(
                np.mean(
                    np.abs(
                        L
                    )
                )
            ),

            measured_tau_abs_mean=float(
                np.mean(
                    np.abs(
                        tau
                    )
                )
            ),

            command_measured_RMSE=float(
                np.sqrt(
                    np.mean(
                        (
                            L
                            -
                            tau
                        )
                        **
                        2
                    )
                )
            ),
        )
    )


print()
print(
    "LOW-LEVEL TORQUE CHECK"
)

display(
    pd.DataFrame(
        command_rows
    )
)


# ======================================================================
# 16. OPTIONAL CHECKPOINT INVENTORY
#
# We do NOT instantiate them because FORCE-5A does not need them.
# We only confirm that the earlier work still exists on Drive.
# ======================================================================

checkpoint_candidates = {

    "V7_PRIOR_SHORT":
    OUT_ROOT /
    "V7_PRIOR_SHORT.pt",

    "V8B_TOUCHDOWN":
    V8_ROOT /
    "B_touchdown" /
    "V8B_TOUCHDOWN_MODEL.pt",

    "V8C_CONTACT_FIELD":
    V8_ROOT /
    "C_contact_field" /
    "V8C_CONTACT_FIELD.pt",

    "V8D_SIAMESE_DELTA":
    V8_ROOT /
    "D_counterfactual" /
    "V8D_SIAMESE_DELTA.pt",

    "V9_C1_DYN":
    V9_C_ROOT /
    "C1_event_joint_residual" /
    "V9C1_DYN.pt",

    "V9_C1_BELIEF":
    V9_C_ROOT /
    "C1_event_joint_residual" /
    "V9C1_BELIEF.pt",

    "V9_C2":
    V9_C_ROOT /
    "C2_structured_contact_posterior" /
    "V9C2_STRUCTURED_POSTERIOR.pt",
}


checkpoint_rows = []


for name, p in checkpoint_candidates.items():

    checkpoint_rows.append(
        dict(
            checkpoint=name,

            exists=p.exists(),

            path=str(
                p
            ),

            size_MB=(
                round(
                    p.stat().st_size
                    /
                    1024**2,
                    2
                )
                if p.exists()
                else
                np.nan
            ),
        )
    )


print()
print(
    "PERSISTENT CHECKPOINT INVENTORY"
)

display(
    pd.DataFrame(
        checkpoint_rows
    )
)


# ======================================================================
# 17. FINAL FORCE-5 PREFLIGHT
# ======================================================================

required_live_objects = {

    "run_records":
    (
        isinstance(
            run_records,
            list
        )
        and
        len(
            run_records
        )
        ==
        4
    ),

    "ANYMAL_D_FK":
    (
        "ANYMAL_D_FK"
        in globals()
    ),

    "DEVICE":
    (
        DEVICE.type
        ==
        "cuda"
    ),

    "PERSIST_ROOT":
    PERSIST_ROOT.exists(),

    "CACHE_ROOT":
    CACHE_ROOT.exists(),

    "V9_ROOT":
    V9_ROOT.exists(),

    "V9_C_ROOT":
    V9_C_ROOT.exists(),
}


print()
print("=" * 112)
print("FORCE-5 FRESH-RUNTIME PREFLIGHT")
print("=" * 112)


for name, ok in required_live_objects.items():

    print(
        (
            "✅"
            if ok
            else
            "❌"
        ),
        name
    )


if not all(
    required_live_objects.values()
):

    raise RuntimeError(
        "Fresh-runtime FORCE-5 preflight failed."
    )


print()
print(
    "GPU allocated:",
    round(
        torch.cuda.memory_allocated()
        /
        1024**3,
        3
    ),
    "GB"
)

print(
    "GPU reserved :",
    round(
        torch.cuda.memory_reserved()
        /
        1024**3,
        3
    ),
    "GB"
)


print()
print("=" * 112)
print("✅ BOOTSTRAP COMPLETE")
print("=" * 112)

print(
    "run_records:",
    len(
        run_records
    )
)

print(
    "physics caches:",
    len(
        physics_files
    )
)

print(
    "visual caches found:",
    sum(
        p.exists()
        for p in visual_files
    ),
    "/",
    len(
        visual_files
    )
)

print()
print(
    "NEXT ACTION:"
)

print(
    "Run the FORCE-5A large cell immediately below this cell."
)

print(
    "Do NOT rerun V7/V8/V9 training."
)

# %% notebook cell 1
# ======================================================================
# FORCE-5A — HARD <5 N PHYSICS CEILING
#
# Goal:
#   Determine whether per-foot 3D contact force can be reconstructed
#   to <= 5 N when future mechanical state is known exactly.
#
# This is NOT yet the causal 1-second forecaster.
#
# Two modes:
#
#   CMD_ORACLE:
#       exact target q,dq,ddq, body/contact configuration
#       + low-level commanded torque
#
#   MEAS_ORACLE:
#       same + exact measured joint torque
#
# Neither receives target contact force as an input.
#
# Physics:
#
#       tau = tau_internal + J^T F
#
# Learn tau_internal from SWING ONLY.
# During stance:
#
#       F_proxy = pinv(J^T) (tau - tau_internal)
#
# Then:
#
#       F_hat = affine_calibration(F_proxy)
#               + bounded neural residual
#
# Strict temporal train/val/test splits.
# ======================================================================

from pathlib import Path
import copy, math, random, time, json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import Ridge


print("=" * 120)
print("FORCE-5A — HARD <5 N PHYSICS CEILING")
print("=" * 120)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_required = [
    "run_records",
    "ANYMAL_D_FK",
    "DEVICE",
]

_missing = [
    x for x in _required
    if x not in globals()
]

if _missing:
    raise RuntimeError(
        "Missing live objects: " + repr(_missing)
    )


if "V9_ROOT" in globals():
    FORCE5_ROOT = (
        Path(V9_ROOT)
        / "V10_FORCE5"
        / "A_physics_ceiling"
    )
else:
    FORCE5_ROOT = (
        Path(PERSIST_ROOT)
        / "V10_FORCE5"
        / "A_physics_ceiling"
    )

FORCE5_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 1. CONFIG
# ======================================================================

SEED5 = 50781

TARGET_N = 5.0

# original data ~100 Hz
DATASET_HOP_SEC = 0.02

# hard temporal blocks
TRAIN_END_FRAC = 0.60

VAL_START_FRAC = 0.65
VAL_END_FRAC = 0.80

TEST_START_FRAC = 0.85


# Command-response lag candidates.
# Uses ONLY present/past command relative to force sample.
CMD_LAGS = [
    0, 1, 2, 3, 4, 5
]


RIDGE_GRID = [
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
]


BATCH = 512

EPOCHS = 120

PATIENCE = 18

LR = 7e-4

WEIGHT_DECAY = 1e-4


RESIDUAL_BLEND_GRID = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
    1.25,
]


LEG_NAMES = [
    "FL",
    "FR",
    "RL",
    "RR",
]


def seed5(s):

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


seed5(SEED5)


# ======================================================================
# 2. VERIFY CACHE
# ======================================================================

for rec in run_records:

    d = rec["data"]

    for k in [
        "time",
        "P",
        "B",
        "U",
        "L_LEG",
        "contact",
    ]:

        if k not in d:

            raise RuntimeError(
                f'{rec["terrain_type"]}: missing {k}'
            )

    if d["P"].shape[1:] != (4, 12):
        raise RuntimeError(
            "Unexpected P shape: " + repr(d["P"].shape)
        )

    if d["L_LEG"].shape[1:] != (4, 3):
        raise RuntimeError(
            "Unexpected L_LEG shape: "
            + repr(d["L_LEG"].shape)
        )


print("✅ cache contract verified")


# ======================================================================
# 3. NUMERICAL JACOBIAN
# ======================================================================

def force5_jacobian(
    q3,
    leg,
    eps=1e-4,
):

    q3 = np.asarray(
        q3,
        np.float64
    )

    q4 = np.zeros(
        (4, 3),
        np.float64
    )

    q4[leg] = q3

    p0 = np.asarray(
        ANYMAL_D_FK
        .foot_positions(q4)[leg],
        np.float64
    )

    J = np.zeros(
        (3, 3),
        np.float64
    )

    for k in range(3):

        qq = q4.copy()

        qq[
            leg,
            k
        ] += eps

        p = np.asarray(
            ANYMAL_D_FK
            .foot_positions(qq)[leg],
            np.float64
        )

        J[:, k] = (
            p - p0
        ) / eps

    return J.astype(
        np.float32
    )


# ======================================================================
# 4. CAUSAL ddq
#
# IMPORTANT:
# backward derivative only.
# No j+1 future derivative.
# ======================================================================

DDQ = {}


for ri, rec in enumerate(
    run_records
):

    d = rec["data"]

    t = np.asarray(
        d["time"],
        np.float64
    )

    dq = np.asarray(
        d["P"][:, :, 3:6],
        np.float64
    )

    ddq = np.zeros_like(dq)

    dt = np.maximum(
        np.diff(t),
        1e-5
    )

    ddq[1:] = (
        dq[1:]
        -
        dq[:-1]
    ) / dt[:, None, None]

    ddq[0] = ddq[1]

    # numerical derivative spike guard
    ddq = np.clip(
        ddq,
        -500.0,
        500.0
    )

    DDQ[ri] = ddq.astype(
        np.float32
    )


print("✅ causal joint accelerations ready")


# ======================================================================
# 5. TEMPORAL SPLITS
# ======================================================================

SPLIT_LIMITS = {}


for ri, rec in enumerate(
    run_records
):

    t = np.asarray(
        rec["data"]["time"],
        np.float64
    )

    t0 = float(t[0])

    T = float(
        t[-1] - t[0]
    )

    SPLIT_LIMITS[ri] = dict(

        train_end=(
            t0
            +
            TRAIN_END_FRAC * T
        ),

        val_start=(
            t0
            +
            VAL_START_FRAC * T
        ),

        val_end=(
            t0
            +
            VAL_END_FRAC * T
        ),

        test_start=(
            t0
            +
            TEST_START_FRAC * T
        ),
    )


def force5_split(
    ri,
    t
):

    lim = SPLIT_LIMITS[ri]

    if t <= lim["train_end"]:
        return "train"

    if (
        t >= lim["val_start"]
        and
        t <= lim["val_end"]
    ):
        return "val"

    if t >= lim["test_start"]:
        return "test"

    return None


# ======================================================================
# 6. INTERNAL-TORQUE FEATURE
#
# Used ONLY to learn robot's non-contact mechanics from SWING samples.
#
# tau_internal =
#     f(q,dq,ddq,body,command,leg)
# ======================================================================

def internal_feature(
    rec,
    ri,
    j,
    leg,
):

    d = rec["data"]

    q = np.asarray(
        d["P"][
            j,
            leg,
            0:3
        ],
        np.float32
    )

    dq = np.asarray(
        d["P"][
            j,
            leg,
            3:6
        ],
        np.float32
    )

    ddq = DDQ[
        ri
    ][
        j,
        leg
    ]

    B = np.asarray(
        d["B"][j],
        np.float32
    )

    U = np.asarray(
        d["U"][j],
        np.float32
    )

    leg_onehot = np.eye(
        4,
        dtype=np.float32
    )[leg]

    return np.concatenate(
        [
            q,
            dq,
            ddq,

            np.sin(q),
            np.cos(q),

            B,
            U,

            leg_onehot,
        ]
    ).astype(
        np.float32
    )


# ======================================================================
# 7. BUILD SWING DATASET FOR tau_internal
# ======================================================================

SW_X = []
SW_Y = []
SW_SPLIT = []


print()
print("Building swing internal-dynamics dataset...")


for ri, rec in enumerate(
    run_records
):

    d = rec["data"]

    t = np.asarray(
        d["time"],
        np.float64
    )

    dt = float(
        np.median(
            np.diff(t)
        )
    )

    stride = max(
        1,
        int(
            round(
                DATASET_HOP_SEC
                /
                dt
            )
        )
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for j in range(
        2,
        len(t),
        stride
    ):

        split = force5_split(
            ri,
            float(t[j])
        )

        if split is None:
            continue

        for leg in range(4):

            # SWING ONLY
            if (
                d["contact"][
                    j,
                    leg
                ]
                >=
                0.5
            ):
                continue

            x = internal_feature(
                rec,
                ri,
                j,
                leg
            )

            y = np.asarray(
                d["P"][
                    j,
                    leg,
                    6:9
                ],
                np.float32
            )

            if not (
                np.isfinite(x).all()
                and
                np.isfinite(y).all()
            ):
                continue

            SW_X.append(x)

            SW_Y.append(y)

            SW_SPLIT.append(split)

            counts[split] += 1

    print(
        rec["terrain_type"],
        counts
    )


SW_X = np.asarray(
    SW_X,
    np.float32
)

SW_Y = np.asarray(
    SW_Y,
    np.float32
)

SW_SPLIT = np.asarray(
    SW_SPLIT
)


sw_train = np.flatnonzero(
    SW_SPLIT == "train"
)

sw_val = np.flatnonzero(
    SW_SPLIT == "val"
)


print(
    "Swing train:",
    len(sw_train),
    "val:",
    len(sw_val)
)


# ======================================================================
# 8. TRAIN-ONLY NORMALIZATION FOR INTERNAL DYNAMICS
# ======================================================================

SW_MEAN = SW_X[
    sw_train
].mean(
    axis=0
).astype(
    np.float32
)

SW_STD = np.maximum(
    SW_X[
        sw_train
    ].std(
        axis=0
    ),
    1e-4
).astype(
    np.float32
)


SW_XN = (
    (
        SW_X
        -
        SW_MEAN
    )
    /
    SW_STD
).astype(
    np.float32
)


# ======================================================================
# 9. SELECT INTERNAL-DYNAMICS RIDGE
# ======================================================================

best_internal = None

best_internal_alpha = None

best_internal_val = np.inf

internal_rows = []


for alpha in RIDGE_GRID:

    model = Ridge(
        alpha=float(alpha)
    )

    model.fit(
        SW_XN[
            sw_train
        ],
        SW_Y[
            sw_train
        ]
    )

    p = model.predict(
        SW_XN[
            sw_val
        ]
    )

    e = np.linalg.norm(
        p
        -
        SW_Y[
            sw_val
        ],
        axis=1
    )

    m = float(
        e.mean()
    )

    internal_rows.append(
        dict(
            alpha=alpha,

            mean_tau_vector_error_Nm=m,

            p95_tau_vector_error_Nm=float(
                np.quantile(
                    e,
                    0.95
                )
            ),
        )
    )

    if m < best_internal_val:

        best_internal_val = m

        best_internal_alpha = alpha

        best_internal = copy.deepcopy(
            model
        )


print()
print(
    "Internal dynamics alpha:",
    best_internal_alpha
)

print(
    "Internal dynamics val mean error:",
    best_internal_val,
    "Nm"
)


# ======================================================================
# 10. HELPER — internal torque at arbitrary sample
# ======================================================================

def predict_internal_tau(
    rec,
    ri,
    j,
    leg,
):

    x = internal_feature(
        rec,
        ri,
        j,
        leg
    )

    xn = (
        x
        -
        SW_MEAN
    ) / SW_STD

    return (
        best_internal
        .predict(
            xn[None]
        )[0]
        .astype(
            np.float32
        )
    )


# ======================================================================
# 11. BUILD LOADED-CONTACT DATASET
#
# IMPORTANT:
# target force is NEVER placed in X.
# ======================================================================

ROWS = []

BASE_FEATURES = []

MEAS_TAU_ALL = []

CMD_LAG_ALL = []

PROXY_OPERATOR = []

TAU_INTERNAL = []

Y_FORCE = []


print()
print(
    "Building loaded-contact force dataset..."
)


t_build = time.time()


for ri, rec in enumerate(
    run_records
):

    d = rec["data"]

    t = np.asarray(
        d["time"],
        np.float64
    )

    dt = float(
        np.median(
            np.diff(t)
        )
    )

    stride = max(
        1,
        int(
            round(
                DATASET_HOP_SEC
                /
                dt
            )
        )
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for j in range(
        max(CMD_LAGS) + 2,
        len(t),
        stride
    ):

        split = force5_split(
            ri,
            float(t[j])
        )

        if split is None:
            continue

        Pj = np.asarray(
            d["P"][j],
            np.float32
        )

        q_all = Pj[:, 0:3]

        dq_all = Pj[:, 3:6]

        ddq_all = DDQ[
            ri
        ][j]

        tau_meas_all = Pj[
            :,
            6:9
        ]

        B = np.asarray(
            d["B"][j],
            np.float32
        )

        U = np.asarray(
            d["U"][j],
            np.float32
        )

        C = np.asarray(
            d["contact"][j],
            np.float32
        )

        foot_all = (
            ANYMAL_D_FK
            .foot_positions(
                q_all
            )
            .astype(
                np.float32
            )
        )

        commands = np.stack(
            [
                np.asarray(
                    d["L_LEG"][
                        j - lag
                    ],
                    np.float32
                )
                for lag in CMD_LAGS
            ],
            axis=0
        )

        for leg in range(4):

            if C[leg] < 0.5:
                continue

            q = q_all[leg]

            dq = dq_all[leg]

            ddq = ddq_all[leg]

            J = force5_jacobian(
                q,
                leg
            )

            # maps torque contribution -> Cartesian force proxy
            JTOP = np.linalg.pinv(
                J.T,
                rcond=1e-4
            ).astype(
                np.float32
            )

            foot = foot_all[leg]

            foot_vel = (
                J
                @
                dq
            ).astype(
                np.float32
            )

            tau_internal = (
                predict_internal_tau(
                    rec,
                    ri,
                    j,
                    leg
                )
            )

            leg_onehot = np.eye(
                4,
                dtype=np.float32
            )[leg]

            # ------------------------------------------------------
            # No contact force anywhere in this feature.
            # ------------------------------------------------------

            base = np.concatenate(
                [
                    q_all.reshape(-1),       # 12
                    dq_all.reshape(-1),      # 12
                    ddq_all.reshape(-1),     # 12

                    B,                       # 12
                    U,                       # 6
                    C,                       # 4

                    q,                       # 3
                    dq,                      # 3
                    ddq,                     # 3

                    np.sin(q),
                    np.cos(q),

                    foot,
                    foot_vel,

                    J.reshape(-1),           # 9

                    tau_internal,            # 3

                    leg_onehot,              # 4

                    np.asarray(
                        [
                            min(
                                math.log1p(
                                    float(
                                        np.linalg.cond(J)
                                    )
                                ),
                                15.0
                            )
                        ],
                        np.float32
                    ),
                ]
            ).astype(
                np.float32
            )

            y = np.asarray(
                Pj[
                    leg,
                    9:12
                ],
                np.float32
            )

            if not (
                np.isfinite(base).all()
                and
                np.isfinite(y).all()
                and
                np.isfinite(JTOP).all()
            ):
                continue

            BASE_FEATURES.append(
                base
            )

            MEAS_TAU_ALL.append(
                tau_meas_all.reshape(-1)
            )

            CMD_LAG_ALL.append(
                commands.reshape(
                    len(CMD_LAGS),
                    -1
                )
            )

            PROXY_OPERATOR.append(
                JTOP
            )

            TAU_INTERNAL.append(
                tau_internal
            )

            Y_FORCE.append(
                y
            )

            ROWS.append(
                dict(
                    ri=ri,

                    terrain_type=rec[
                        "terrain_type"
                    ],

                    environment=rec[
                        "environment"
                    ],

                    run=rec[
                        "run"
                    ],

                    split=split,

                    time=float(
                        t[j]
                    ),

                    index=int(j),

                    leg=int(leg),

                    leg_name=LEG_NAMES[
                        leg
                    ],
                )
            )

            counts[split] += 1

    print(
        rec["terrain_type"],
        counts
    )


BASE_FEATURES = np.asarray(
    BASE_FEATURES,
    np.float32
)

MEAS_TAU_ALL = np.asarray(
    MEAS_TAU_ALL,
    np.float32
)

CMD_LAG_ALL = np.asarray(
    CMD_LAG_ALL,
    np.float32
)

PROXY_OPERATOR = np.asarray(
    PROXY_OPERATOR,
    np.float32
)

TAU_INTERNAL = np.asarray(
    TAU_INTERNAL,
    np.float32
)

Y_FORCE = np.asarray(
    Y_FORCE,
    np.float32
)

force5_meta = pd.DataFrame(
    ROWS
)


print()
print(
    "Dataset build time:",
    time.time() - t_build,
    "s"
)

print(
    "Loaded samples:",
    len(
        Y_FORCE
    )
)


train_ids = np.flatnonzero(
    force5_meta.split.to_numpy()
    ==
    "train"
)

val_ids = np.flatnonzero(
    force5_meta.split.to_numpy()
    ==
    "val"
)

test_ids = np.flatnonzero(
    force5_meta.split.to_numpy()
    ==
    "test"
)


print(
    "TRAIN:",
    len(train_ids)
)

print(
    "VAL  :",
    len(val_ids)
)

print(
    "TEST :",
    len(test_ids)
)


if min(
    len(train_ids),
    len(val_ids),
    len(test_ids)
) == 0:

    raise RuntimeError(
        "Temporal split produced empty subset."
    )


# ======================================================================
# 12. PHYSICS PROXY
# ======================================================================

def make_proxy(
    tau_source
):

    tau_source = np.asarray(
        tau_source,
        np.float32
    )

    dtau = (
        tau_source
        -
        TAU_INTERNAL
    )

    return np.einsum(
        "nij,nj->ni",
        PROXY_OPERATOR,
        dtau
    ).astype(
        np.float32
    )


# MEASURED ORACLE TARGET-LEG TORQUE
TAU_MEAS_TARGET = np.stack(
    [
        MEAS_TAU_ALL[
            i,
            3 * force5_meta.leg.iloc[i]:
            3 * force5_meta.leg.iloc[i] + 3
        ]

        for i in range(
            len(force5_meta)
        )
    ]
).astype(
    np.float32
)


PROXY_MEAS_RAW = make_proxy(
    TAU_MEAS_TARGET
)


# ======================================================================
# 13. PER-LEG AFFINE PHYSICS CALIBRATION
#
# Learns frame/sign/scale convention ONLY on train data.
# ======================================================================

def affine_calibrate_proxy(
    proxy
):

    out = np.zeros_like(
        proxy,
        dtype=np.float32
    )

    models = {}

    for leg in range(4):

        tr = train_ids[
            force5_meta.iloc[
                train_ids
            ].leg.to_numpy()
            ==
            leg
        ]

        if len(tr) < 20:
            raise RuntimeError(
                f"Too few train samples for leg {leg}"
            )

        model = Ridge(
            alpha=1e-3
        )

        model.fit(
            proxy[tr],
            Y_FORCE[tr]
        )

        out[:]  # keep shape alive

        ids = np.flatnonzero(
            force5_meta.leg.to_numpy()
            ==
            leg
        )

        out[
            ids
        ] = model.predict(
            proxy[
                ids
            ]
        ).astype(
            np.float32
        )

        models[leg] = model

    return (
        out,
        models
    )


PROXY_MEAS, MEAS_CAL_MODELS = (
    affine_calibrate_proxy(
        PROXY_MEAS_RAW
    )
)


# ======================================================================
# 14. COMMAND LAG SEARCH ON VALIDATION ONLY
# ======================================================================

cmd_lag_rows = []

BEST_CMD_LAG = None

BEST_CMD_PROXY = None

BEST_CMD_CAL_MODELS = None

best_cmd_val = np.inf


for lag_i, lag in enumerate(
    CMD_LAGS
):

    tau_cmd_target = np.stack(
        [
            CMD_LAG_ALL[
                i,
                lag_i,
                3 * force5_meta.leg.iloc[i]:
                3 * force5_meta.leg.iloc[i] + 3
            ]

            for i in range(
                len(force5_meta)
            )
        ]
    ).astype(
        np.float32
    )

    raw = make_proxy(
        tau_cmd_target
    )

    calibrated, models = (
        affine_calibrate_proxy(
            raw
        )
    )

    err = np.linalg.norm(
        calibrated[
            val_ids
        ]
        -
        Y_FORCE[
            val_ids
        ],
        axis=1
    )

    vm = float(
        err.mean()
    )

    cmd_lag_rows.append(
        dict(
            command_lag_samples=lag,

            val_mean_vector_error_N=vm,

            val_p95_vector_error_N=float(
                np.quantile(
                    err,
                    0.95
                )
            ),
        )
    )

    if vm < best_cmd_val:

        best_cmd_val = vm

        BEST_CMD_LAG = lag

        BEST_CMD_PROXY = (
            calibrated.copy()
        )

        BEST_CMD_CAL_MODELS = models


cmd_lag_table = pd.DataFrame(
    cmd_lag_rows
)


print()
print(
    "Validation-selected command lag:",
    BEST_CMD_LAG,
    "samples"
)

display(
    cmd_lag_table
)


# ======================================================================
# 15. RESIDUAL NETWORK
#
# Learns ONLY the remaining error around physics reconstruction.
# ======================================================================

class Force5ResidualNet(
    nn.Module
):

    def __init__(
        self,
        din
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.LayerNorm(
                din
            ),

            nn.Linear(
                din,
                384
            ),

            nn.GELU(),

            nn.Dropout(
                0.05
            ),

            nn.Linear(
                384,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.05
            ),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                3
            ),
        )

        # starts as pure physics model
        nn.init.zeros_(
            self.net[-1].weight
        )

        nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(
        self,
        x
    ):

        return self.net(x)


# ======================================================================
# 16. FIT ONE MODE
# ======================================================================

def train_force5_mode(
    mode_name,
    physics_proxy,
    source_tau_all,
    seed_offset,
):

    print()
    print("=" * 100)
    print("TRAINING", mode_name)
    print("=" * 100)

    # --------------------------------------------------------------
    # Residual feature.
    # --------------------------------------------------------------

    X = np.concatenate(
        [
            BASE_FEATURES,

            source_tau_all,

            physics_proxy,

            np.linalg.norm(
                physics_proxy,
                axis=1,
                keepdims=True
            ).astype(
                np.float32
            ),
        ],
        axis=1
    ).astype(
        np.float32
    )

    Xmean = X[
        train_ids
    ].mean(
        axis=0
    ).astype(
        np.float32
    )

    Xstd = np.maximum(
        X[
            train_ids
        ].std(
            axis=0
        ),
        1e-4
    ).astype(
        np.float32
    )

    Xn = (
        (
            X
            -
            Xmean
        )
        /
        Xstd
    ).astype(
        np.float32
    )

    residual = (
        Y_FORCE
        -
        physics_proxy
    ).astype(
        np.float32
    )

    Rstd = np.maximum(
        residual[
            train_ids
        ].std(
            axis=0
        ),
        1.0
    ).astype(
        np.float32
    )

    train_ds = TensorDataset(

        torch.from_numpy(
            Xn[
                train_ids
            ]
        ),

        torch.from_numpy(
            residual[
                train_ids
            ]
        ),
    )

    train_dl = DataLoader(

        train_ds,

        batch_size=BATCH,

        shuffle=True,

        num_workers=0,
    )

    model = (
        Force5ResidualNet(
            X.shape[1]
        )
        .to(
            DEVICE
        )
    )

    seed5(
        SEED5
        +
        seed_offset
    )

    opt = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=(
            WEIGHT_DECAY
        ),
    )

    rstd_t = torch.as_tensor(

        Rstd,

        dtype=torch.float32,

        device=DEVICE,
    )

    Xval_t = torch.as_tensor(

        Xn[
            val_ids
        ],

        dtype=torch.float32,

        device=DEVICE,
    )

    best_state = None

    best_val = np.inf

    best_epoch = None

    patience = 0

    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        tloss = []

        for xb, yb in train_dl:

            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            opt.zero_grad(
                set_to_none=True
            )

            pred = (
                model(xb)
                *
                rstd_t[
                    None
                ]
            )

            vector_loss = (
                torch.linalg.vector_norm(
                    pred
                    -
                    yb,
                    dim=-1
                )
                .mean()
            )

            component_loss = (
                Fnn.smooth_l1_loss(
                    pred,
                    yb,
                    beta=3.0
                )
            )

            loss = (
                vector_loss
                +
                0.25
                *
                component_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            opt.step()

            tloss.append(
                float(
                    loss.detach().cpu()
                )
            )

        # ----------------------------------------------------------
        # Validation in physical NEWTONS.
        # ----------------------------------------------------------

        model.eval()

        with torch.inference_mode():

            r = (
                model(
                    Xval_t
                )
                *
                rstd_t[
                    None
                ]
            )

            r = (
                r.cpu()
                .numpy()
            )

        pred_val = (
            physics_proxy[
                val_ids
            ]
            +
            r
        )

        err = np.linalg.norm(
            pred_val
            -
            Y_FORCE[
                val_ids
            ],
            axis=1
        )

        vm = float(
            err.mean()
        )

        vp95 = float(
            np.quantile(
                err,
                0.95
            )
        )

        history.append(
            dict(
                epoch=epoch,

                train_loss=float(
                    np.mean(
                        tloss
                    )
                ),

                val_mean_vector_error_N=vm,

                val_p95_vector_error_N=vp95,
            )
        )

        improved = (
            vm
            <
            best_val
            -
            1e-3
        )

        if improved:

            best_val = vm

            best_epoch = epoch

            best_state = {
                k:
                v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()
            }

            patience = 0

        else:

            patience += 1

        if (
            epoch == 1
            or
            epoch % 10 == 0
            or
            improved
        ):

            print(
                f"{mode_name:12s} "
                f"ep={epoch:03d} "
                f"train={np.mean(tloss):.3f} "
                f"val_mean={vm:.3f} N "
                f"val_p95={vp95:.3f} N"
            )

        if patience >= PATIENCE:

            print(
                "early stop at",
                epoch
            )

            break


    model.load_state_dict(
        best_state
    )

    model.eval()


    # --------------------------------------------------------------
    # Validation-select residual strength.
    # --------------------------------------------------------------

    with torch.inference_mode():

        rv = (
            model(
                Xval_t
            )
            *
            rstd_t[
                None
            ]
        )

        rv = (
            rv.cpu()
            .numpy()
        )


    best_lambda = None

    best_lambda_val = np.inf

    lambda_rows = []


    for lam in RESIDUAL_BLEND_GRID:

        p = (
            physics_proxy[
                val_ids
            ]
            +
            float(lam)
            *
            rv
        )

        e = np.linalg.norm(
            p
            -
            Y_FORCE[
                val_ids
            ],
            axis=1
        )

        m = float(
            e.mean()
        )

        lambda_rows.append(
            dict(
                blend_lambda=lam,

                val_mean_vector_error_N=m,

                val_p95_vector_error_N=float(
                    np.quantile(
                        e,
                        0.95
                    )
                ),
            )
        )

        if m < best_lambda_val:

            best_lambda_val = m

            best_lambda = float(lam)


    print(
        "best residual lambda:",
        best_lambda
    )

    print(
        "best validation vector error:",
        best_lambda_val,
        "N"
    )


    return dict(

        name=mode_name,

        model=model,

        Xmean=Xmean,

        Xstd=Xstd,

        Rstd=Rstd,

        physics_proxy=physics_proxy,

        source_tau_all=source_tau_all,

        lambda_=best_lambda,

        history=pd.DataFrame(
            history
        ),

        blend_table=pd.DataFrame(
            lambda_rows
        ),
    )


# ======================================================================
# 17. BUILD COMMAND SOURCE USING SELECTED LAG
# ======================================================================

best_lag_i = CMD_LAGS.index(
    BEST_CMD_LAG
)


CMD_SOURCE_ALL = CMD_LAG_ALL[
    :,
    best_lag_i,
    :
].astype(
    np.float32
)


MEAS_SOURCE_ALL = MEAS_TAU_ALL.astype(
    np.float32
)


# ======================================================================
# 18. TRAIN BOTH CEILINGS
# ======================================================================

CMD_BUNDLE = train_force5_mode(

    "CMD_ORACLE",

    BEST_CMD_PROXY,

    CMD_SOURCE_ALL,

    100,
)


MEAS_BUNDLE = train_force5_mode(

    "MEAS_ORACLE",

    PROXY_MEAS,

    MEAS_SOURCE_ALL,

    200,
)


# ======================================================================
# 19. PREDICT
# ======================================================================

@torch.inference_mode()
def predict_bundle(
    bundle,
    ids,
):

    ids = np.asarray(
        ids,
        np.int64
    )

    X = np.concatenate(
        [
            BASE_FEATURES[
                ids
            ],

            bundle[
                "source_tau_all"
            ][
                ids
            ],

            bundle[
                "physics_proxy"
            ][
                ids
            ],

            np.linalg.norm(
                bundle[
                    "physics_proxy"
                ][
                    ids
                ],
                axis=1,
                keepdims=True
            ).astype(
                np.float32
            ),
        ],
        axis=1
    ).astype(
        np.float32
    )

    Xn = (
        (
            X
            -
            bundle[
                "Xmean"
            ]
        )
        /
        bundle[
            "Xstd"
        ]
    ).astype(
        np.float32
    )

    preds = []

    bs = 2048

    rstd_t = torch.as_tensor(

        bundle[
            "Rstd"
        ],

        dtype=torch.float32,

        device=DEVICE,
    )

    for st in range(
        0,
        len(ids),
        bs
    ):

        xb = torch.as_tensor(

            Xn[
                st:
                st + bs
            ],

            dtype=torch.float32,

            device=DEVICE,
        )

        r = (
            bundle[
                "model"
            ](
                xb
            )
            *
            rstd_t[
                None
            ]
        )

        preds.append(
            r.cpu()
            .numpy()
        )

    residual = np.concatenate(
        preds,
        axis=0
    )

    final = (
        bundle[
            "physics_proxy"
        ][
            ids
        ]
        +
        bundle[
            "lambda_"
        ]
        *
        residual
    )

    return final.astype(
        np.float32
    )


P_CMD = predict_bundle(
    CMD_BUNDLE,
    test_ids
)


P_MEAS = predict_bundle(
    MEAS_BUNDLE,
    test_ids
)


GT = Y_FORCE[
    test_ids
]


E_CMD = np.linalg.norm(
    P_CMD
    -
    GT,
    axis=1
)


E_MEAS = np.linalg.norm(
    P_MEAS
    -
    GT,
    axis=1
)


# ======================================================================
# 20. METRICS
# ======================================================================

def force_metrics(
    e
):

    e = np.asarray(
        e,
        np.float64
    )

    return dict(

        samples=len(e),

        mean_vector_error_N=float(
            e.mean()
        ),

        median_vector_error_N=float(
            np.median(e)
        ),

        p90_vector_error_N=float(
            np.quantile(
                e,
                0.90
            )
        ),

        p95_vector_error_N=float(
            np.quantile(
                e,
                0.95
            )
        ),

        p99_vector_error_N=float(
            np.quantile(
                e,
                0.99
            )
        ),

        max_vector_error_N=float(
            e.max()
        ),

        within_5N_rate=float(
            np.mean(
                e <= TARGET_N
            )
        ),

        p95_under_5N=bool(
            np.quantile(
                e,
                0.95
            )
            <= TARGET_N
        ),

        literal_max_under_5N=bool(
            e.max()
            <= TARGET_N
        ),
    )


summary = pd.DataFrame(
    [
        dict(
            mode="CMD_ORACLE",
            **force_metrics(
                E_CMD
            )
        ),

        dict(
            mode="MEAS_ORACLE",
            **force_metrics(
                E_MEAS
            )
        ),
    ]
)


# ======================================================================
# 21. PER FOOT / TERRAIN
# ======================================================================

test_meta = (
    force5_meta
    .iloc[
        test_ids
    ]
    .reset_index(
        drop=True
    )
)


test_raw = test_meta.copy()


test_raw[
    "cmd_error_N"
] = E_CMD


test_raw[
    "meas_error_N"
] = E_MEAS


for k, ax in enumerate(
    ["x", "y", "z"]
):

    test_raw[
        f"gt_F{ax}_N"
    ] = GT[:, k]

    test_raw[
        f"cmd_F{ax}_N"
    ] = P_CMD[:, k]

    test_raw[
        f"meas_F{ax}_N"
    ] = P_MEAS[:, k]


group_rows = []


for (
    terrain,
    leg
), g in test_raw.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    group_rows.append(
        dict(
            terrain_type=terrain,

            leg_name=leg,

            mode="CMD_ORACLE",

            **force_metrics(
                g[
                    "cmd_error_N"
                ]
            )
        )
    )

    group_rows.append(
        dict(
            terrain_type=terrain,

            leg_name=leg,

            mode="MEAS_ORACLE",

            **force_metrics(
                g[
                    "meas_error_N"
                ]
            )
        )
    )


by_terrain_leg = pd.DataFrame(
    group_rows
)


# ======================================================================
# 22. RAW FORCE VARIATION AUDIT
#
# Shows how violently labels themselves move between adjacent samples.
# ======================================================================

variation_rows = []


for rec in run_records:

    d = rec["data"]

    forces = np.asarray(
        d["P"][
            :,
            :,
            9:12
        ],
        np.float32
    )

    contact = (
        np.asarray(
            d["contact"]
        )
        >=
        0.5
    )

    for leg in range(4):

        valid = (
            contact[
                1:,
                leg
            ]
            &
            contact[
                :-1,
                leg
            ]
        )

        delta = np.linalg.norm(

            forces[
                1:,
                leg
            ]
            -
            forces[
                :-1,
                leg
            ],

            axis=1
        )

        delta = delta[
            valid
        ]

        if len(delta) == 0:
            continue

        variation_rows.append(
            dict(
                terrain_type=rec[
                    "terrain_type"
                ],

                leg_name=LEG_NAMES[
                    leg
                ],

                n=len(delta),

                median_adjacent_change_N=float(
                    np.median(
                        delta
                    )
                ),

                p95_adjacent_change_N=float(
                    np.quantile(
                        delta,
                        0.95
                    )
                ),

                max_adjacent_change_N=float(
                    delta.max()
                ),
            )
        )


force_variation = pd.DataFrame(
    variation_rows
)


# ======================================================================
# 23. SAVE
# ======================================================================

summary.to_csv(

    FORCE5_ROOT /
    "FORCE5A_SUMMARY.csv",

    index=False
)


by_terrain_leg.to_csv(

    FORCE5_ROOT /
    "FORCE5A_BY_TERRAIN_LEG.csv",

    index=False
)


test_raw.to_csv(

    FORCE5_ROOT /
    "FORCE5A_STRICT_TEST_RAW.csv",

    index=False
)


force_variation.to_csv(

    FORCE5_ROOT /
    "FORCE5A_RAW_FORCE_VARIATION.csv",

    index=False
)


cmd_lag_table.to_csv(

    FORCE5_ROOT /
    "FORCE5A_COMMAND_LAG_SEARCH.csv",

    index=False
)


pd.DataFrame(
    internal_rows
).to_csv(

    FORCE5_ROOT /
    "FORCE5A_INTERNAL_DYNAMICS_SEARCH.csv",

    index=False
)


CMD_BUNDLE[
    "history"
].to_csv(

    FORCE5_ROOT /
    "CMD_ORACLE_HISTORY.csv",

    index=False
)


MEAS_BUNDLE[
    "history"
].to_csv(

    FORCE5_ROOT /
    "MEAS_ORACLE_HISTORY.csv",

    index=False
)


CMD_BUNDLE[
    "blend_table"
].to_csv(

    FORCE5_ROOT /
    "CMD_ORACLE_BLEND.csv",

    index=False
)


MEAS_BUNDLE[
    "blend_table"
].to_csv(

    FORCE5_ROOT /
    "MEAS_ORACLE_BLEND.csv",

    index=False
)


# ======================================================================
# 24. OUTPUT
# ======================================================================

print()
print("=" * 120)
print("FORCE LABEL ADJACENT-SAMPLE VARIATION")
print("=" * 120)

display(
    force_variation
)


print()
print("=" * 120)
print("FORCE-5A STRICT TEST")
print("=" * 120)

display(
    summary
)


print()
print("=" * 120)
print("FORCE-5A BY TERRAIN / FOOT")
print("=" * 120)

display(
    by_terrain_leg
)


cmd = summary[
    summary.mode
    ==
    "CMD_ORACLE"
].iloc[0]


meas = summary[
    summary.mode
    ==
    "MEAS_ORACLE"
].iloc[0]


print()
print("=" * 120)
print("FORCE-5A HARD DECISION")
print("=" * 120)


print(
    "CMD_ORACLE:"
)

print(
    "  mean   =",
    cmd.mean_vector_error_N,
    "N"
)

print(
    "  p95    =",
    cmd.p95_vector_error_N,
    "N"
)

print(
    "  max    =",
    cmd.max_vector_error_N,
    "N"
)

print(
    "  <=5 N  =",
    100.0
    *
    cmd.within_5N_rate,
    "%"
)


print()
print(
    "MEAS_ORACLE:"
)

print(
    "  mean   =",
    meas.mean_vector_error_N,
    "N"
)

print(
    "  p95    =",
    meas.p95_vector_error_N,
    "N"
)

print(
    "  max    =",
    meas.max_vector_error_N,
    "N"
)

print(
    "  <=5 N  =",
    100.0
    *
    meas.within_5N_rate,
    "%"
)


print()


if (
    meas.p95_vector_error_N
    <=
    TARGET_N
):

    FORCE5A_STATUS = (
        "5N_RECONSTRUCTIBLE"
    )

    print(
        "✅ 5 N PHYSICS CEILING ACHIEVED."
    )

    if (
        cmd.p95_vector_error_N
        <=
        TARGET_N
    ):

        print(
            "Even commanded torque is sufficient."
        )

        print(
            "Next: FORCE-5B 1-second causal "
            "state + force trajectory Transformer."
        )

    else:

        print(
            "Measured torque is sufficient but "
            "command torque is not."
        )

        print(
            "Next: forecast actuator response first, "
            "then decode forces."
        )


elif (
    meas.p95_vector_error_N
    <=
    10.0
):

    FORCE5A_STATUS = (
        "NEAR_5N"
    )

    print(
        "🟡 WE ARE CLOSE TO THE PHYSICAL CEILING."
    )

    print(
        "Next: timestamp / force-frame / "
        "Jacobian-system-identification refinement."
    )


else:

    FORCE5A_STATUS = (
        "NOT_RECONSTRUCTIBLE_YET"
    )

    print(
        "❌ DO NOT BUILD THE 1-SECOND TRANSFORMER YET."
    )

    print(
        "Even exact mechanical state does not "
        "reconstruct the force near 5 N."
    )

    print(
        "We must fix synchronization, force-frame "
        "convention, dynamics identification, or label quality."
    )


print()
print(
    "STATUS:",
    FORCE5A_STATUS
)

print()
print(
    "Saved:",
    FORCE5_ROOT
)

print(
    "✅ FORCE-5A COMPLETE"
)

# %% notebook cell 2
# ======================================================================
# FORCE-5A DIAGNOSTIC PATCH
#
# 1. Fixes KeyError: False
# 2. Finishes FORCE-5A reporting without retraining
# 3. Selects exactly 4 reproducible random test cases:
#       one from each terrain
# 4. Prints ALL available per-leg sensor/state values
# 5. Prints exact force reconstruction results for each stance leg
#
# RUN AFTER THE FAILED FORCE-5A CELL.
# ======================================================================

import numpy as np
import pandas as pd
import json
from pathlib import Path

from IPython.display import display


print("=" * 120)
print("FORCE-5A PATCH + FOUR RANDOM TERRAIN TEST CASES")
print("=" * 120)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_required = [
    "summary",
    "test_raw",
    "force5_meta",
    "run_records",
    "ANYMAL_D_FK",
    "DDQ",
    "TARGET_N",
]

_missing = [
    x for x in _required
    if x not in globals()
]

if _missing:

    raise RuntimeError(
        "Missing FORCE-5A objects: "
        +
        repr(_missing)
        +
        "\nThe large FORCE-5A cell must have reached evaluation first."
    )


# ======================================================================
# 1. FIX THE EXACT PANDAS BUG
#
# WRONG:
#
#     summary.mode
#
# because .mode is a DataFrame method.
#
# RIGHT:
#
#     summary["mode"]
# ======================================================================

cmd_rows = summary.loc[
    summary["mode"].eq(
        "CMD_ORACLE"
    )
]

meas_rows = summary.loc[
    summary["mode"].eq(
        "MEAS_ORACLE"
    )
]


if len(cmd_rows) != 1:

    raise RuntimeError(
        "Expected exactly one CMD_ORACLE summary row."
    )


if len(meas_rows) != 1:

    raise RuntimeError(
        "Expected exactly one MEAS_ORACLE summary row."
    )


cmd = cmd_rows.iloc[0]

meas = meas_rows.iloc[0]


print()
print("✅ Pandas reporting bug fixed.")
print()


print("CMD_ORACLE")

print(
    "  mean       =",
    float(
        cmd["mean_vector_error_N"]
    ),
    "N"
)

print(
    "  p95        =",
    float(
        cmd["p95_vector_error_N"]
    ),
    "N"
)

print(
    "  max        =",
    float(
        cmd["max_vector_error_N"]
    ),
    "N"
)

print(
    "  <=5N rate  =",
    100.0
    *
    float(
        cmd["within_5N_rate"]
    ),
    "%"
)


print()
print("MEAS_ORACLE")

print(
    "  mean       =",
    float(
        meas["mean_vector_error_N"]
    ),
    "N"
)

print(
    "  p95        =",
    float(
        meas["p95_vector_error_N"]
    ),
    "N"
)

print(
    "  max        =",
    float(
        meas["max_vector_error_N"]
    ),
    "N"
)

print(
    "  <=5N rate  =",
    100.0
    *
    float(
        meas["within_5N_rate"]
    ),
    "%"
)


# ======================================================================
# 2. RESTORE A CLEAN RESEARCH STATUS
# ======================================================================

if float(
    meas["p95_vector_error_N"]
) <= TARGET_N:

    FORCE5A_STATUS = (
        "5N_RECONSTRUCTIBLE"
    )

elif float(
    meas["p95_vector_error_N"]
) <= 10.0:

    FORCE5A_STATUS = (
        "NEAR_5N"
    )

else:

    FORCE5A_STATUS = (
        "STATIC_PHYSICS_MODEL_FAILED"
    )


print()
print(
    "FORCE5A_STATUS =",
    FORCE5A_STATUS
)


# ======================================================================
# 3. HELPER — NUMERICAL JACOBIAN
#
# Reuse existing force5_jacobian if alive.
# ======================================================================

if "force5_jacobian" not in globals():

    def force5_jacobian(
        q3,
        leg,
        eps=1e-4,
    ):

        q3 = np.asarray(
            q3,
            np.float64
        )

        q4 = np.zeros(
            (
                4,
                3
            ),
            np.float64
        )

        q4[
            int(leg)
        ] = q3

        p0 = np.asarray(
            ANYMAL_D_FK
            .foot_positions(
                q4
            )[
                int(leg)
            ],
            np.float64
        )

        J = np.zeros(
            (
                3,
                3
            ),
            np.float64
        )

        for k in range(3):

            qq = q4.copy()

            qq[
                int(leg),
                k
            ] += eps

            pp = np.asarray(
                ANYMAL_D_FK
                .foot_positions(
                    qq
                )[
                    int(leg)
                ],
                np.float64
            )

            J[
                :,
                k
            ] = (
                pp - p0
            ) / eps

        return J.astype(
            np.float32
        )


# ======================================================================
# 4. CHOOSE EXACTLY ONE RANDOM STRICT-TEST TIME PER TERRAIN
#
# Fixed seed -> reproducible.
#
# Anchor times come from test_raw, meaning they are guaranteed to lie
# inside the strict test region.
# ======================================================================

RANDOM_TEST_SEED = 20260909

rng = np.random.default_rng(
    RANDOM_TEST_SEED
)


terrain_order = [
    "forest",
    "swamp_marsh",
    "rough_uneven",
    "wet_uneven_proxy",
]


selected_cases = []


for terrain in terrain_order:

    q = test_raw.loc[
        test_raw[
            "terrain_type"
        ].eq(
            terrain
        )
    ].copy()

    if len(q) == 0:

        raise RuntimeError(
            f"No strict test rows for {terrain}"
        )

    # Select among unique physical sample indices,
    # not among individual leg rows.
    candidates = (
        q[
            [
                "ri",
                "index",
                "time",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    r = candidates.iloc[
        int(
            rng.integers(
                0,
                len(candidates)
            )
        )
    ]

    selected_cases.append(
        dict(
            terrain_type=terrain,

            ri=int(
                r["ri"]
            ),

            index=int(
                r["index"]
            ),

            time=float(
                r["time"]
            ),
        )
    )


force5_random_cases = pd.DataFrame(
    selected_cases
)


print()
print("=" * 120)
print("FOUR RANDOM STRICT-TEST CASES")
print("=" * 120)

display(
    force5_random_cases
)


# ======================================================================
# 5. COLLECT EVERY LEGWISE PHYSICS/SENSOR PARAMETER
# ======================================================================

LEG_NAMES_PATCH = [
    "FL",
    "FR",
    "RL",
    "RR",
]


sensor_rows = []

global_rows = []

prediction_rows = []


for case_id, c in force5_random_cases.iterrows():

    ri = int(
        c["ri"]
    )

    j = int(
        c["index"]
    )

    rec = run_records[
        ri
    ]

    d = rec[
        "data"
    ]

    P = np.asarray(
        d["P"][j],
        np.float32
    )

    B = np.asarray(
        d["B"][j],
        np.float32
    )

    U = np.asarray(
        d["U"][j],
        np.float32
    )

    L = np.asarray(
        d["L_LEG"][j],
        np.float32
    )

    C = np.asarray(
        d["contact"][j],
        np.float32
    )

    E = (
        np.asarray(
            d["E"][j],
            np.float32
        )
        if "E" in d
        else
        np.full(
            (
                4,
                3
            ),
            np.nan,
            np.float32
        )
    )

    ddq = np.asarray(
        DDQ[
            ri
        ][
            j
        ],
        np.float32
    )

    q4 = P[
        :,
        0:3
    ]

    dq4 = P[
        :,
        3:6
    ]

    feet = (
        ANYMAL_D_FK
        .foot_positions(
            q4
        )
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------------
    # Global body + command values.
    # --------------------------------------------------------------

    gr = dict(

        case_id=int(
            case_id + 1
        ),

        terrain_type=rec[
            "terrain_type"
        ],

        environment=rec[
            "environment"
        ],

        run=rec[
            "run"
        ],

        index=j,

        time_s=float(
            d["time"][j]
        ),
    )


    for k in range(
        len(B)
    ):

        gr[
            f"B{k}"
        ] = float(
            B[k]
        )


    for k in range(
        len(U)
    ):

        gr[
            f"U{k}"
        ] = float(
            U[k]
        )


    global_rows.append(
        gr
    )


    # --------------------------------------------------------------
    # Per leg.
    # --------------------------------------------------------------

    for leg in range(4):

        J = force5_jacobian(
            q4[
                leg
            ],
            leg
        )

        foot_vel = (
            J
            @
            dq4[
                leg
            ]
        )

        F_true = P[
            leg,
            9:12
        ]

        tau = P[
            leg,
            6:9
        ]


        sr = dict(

            case_id=int(
                case_id + 1
            ),

            terrain_type=rec[
                "terrain_type"
            ],

            time_s=float(
                d["time"][j]
            ),

            leg=LEG_NAMES_PATCH[
                leg
            ],

            contact=int(
                C[
                    leg
                ] >= 0.5
            ),

            # joint position
            q1=float(
                P[
                    leg,
                    0
                ]
            ),

            q2=float(
                P[
                    leg,
                    1
                ]
            ),

            q3=float(
                P[
                    leg,
                    2
                ]
            ),

            # velocity
            dq1=float(
                P[
                    leg,
                    3
                ]
            ),

            dq2=float(
                P[
                    leg,
                    4
                ]
            ),

            dq3=float(
                P[
                    leg,
                    5
                ]
            ),

            # acceleration
            ddq1=float(
                ddq[
                    leg,
                    0
                ]
            ),

            ddq2=float(
                ddq[
                    leg,
                    1
                ]
            ),

            ddq3=float(
                ddq[
                    leg,
                    2
                ]
            ),

            # measured torque
            tau1_Nm=float(
                tau[
                    0
                ]
            ),

            tau2_Nm=float(
                tau[
                    1
                ]
            ),

            tau3_Nm=float(
                tau[
                    2
                ]
            ),

            # low-level commanded torque
            cmd_tau1_Nm=float(
                L[
                    leg,
                    0
                ]
            ),

            cmd_tau2_Nm=float(
                L[
                    leg,
                    1
                ]
            ),

            cmd_tau3_Nm=float(
                L[
                    leg,
                    2
                ]
            ),

            # historical command-response error
            E1_Nm=float(
                E[
                    leg,
                    0
                ]
            ),

            E2_Nm=float(
                E[
                    leg,
                    1
                ]
            ),

            E3_Nm=float(
                E[
                    leg,
                    2
                ]
            ),

            # measured contact force
            Fx_N=float(
                F_true[
                    0
                ]
            ),

            Fy_N=float(
                F_true[
                    1
                ]
            ),

            Fz_N=float(
                F_true[
                    2
                ]
            ),

            Fmag_N=float(
                np.linalg.norm(
                    F_true
                )
            ),

            # FK
            foot_x_m=float(
                feet[
                    leg,
                    0
                ]
            ),

            foot_y_m=float(
                feet[
                    leg,
                    1
                ]
            ),

            foot_z_m=float(
                feet[
                    leg,
                    2
                ]
            ),

            foot_vx_mps=float(
                foot_vel[
                    0
                ]
            ),

            foot_vy_mps=float(
                foot_vel[
                    1
                ]
            ),

            foot_vz_mps=float(
                foot_vel[
                    2
                ]
            ),

            jacobian_condition=float(
                np.linalg.cond(
                    J
                )
            ),
        )


        sensor_rows.append(
            sr
        )


        # ----------------------------------------------------------
        # Prediction result at EXACT SAME physical sample.
        #
        # test_raw contains rows only for loaded-contact legs.
        # ----------------------------------------------------------

        m = test_raw.loc[

            test_raw[
                "ri"
            ].eq(
                ri
            )

            &

            test_raw[
                "index"
            ].eq(
                j
            )

            &

            test_raw[
                "leg"
            ].eq(
                leg
            )
        ]


        if len(m):

            rr = m.iloc[
                0
            ]


            pred_row = dict(

                case_id=int(
                    case_id + 1
                ),

                terrain_type=rec[
                    "terrain_type"
                ],

                time_s=float(
                    d["time"][j]
                ),

                leg=LEG_NAMES_PATCH[
                    leg
                ],

                contact=1,

                gt_Fx_N=float(
                    rr[
                        "gt_Fx_N"
                    ]
                ),

                gt_Fy_N=float(
                    rr[
                        "gt_Fy_N"
                    ]
                ),

                gt_Fz_N=float(
                    rr[
                        "gt_Fz_N"
                    ]
                ),

                cmd_Fx_N=float(
                    rr[
                        "cmd_Fx_N"
                    ]
                ),

                cmd_Fy_N=float(
                    rr[
                        "cmd_Fy_N"
                    ]
                ),

                cmd_Fz_N=float(
                    rr[
                        "cmd_Fz_N"
                    ]
                ),

                meas_Fx_N=float(
                    rr[
                        "meas_Fx_N"
                    ]
                ),

                meas_Fy_N=float(
                    rr[
                        "meas_Fy_N"
                    ]
                ),

                meas_Fz_N=float(
                    rr[
                        "meas_Fz_N"
                    ]
                ),

                cmd_error_N=float(
                    rr[
                        "cmd_error_N"
                    ]
                ),

                meas_error_N=float(
                    rr[
                        "meas_error_N"
                    ]
                ),

                meas_error_over_5N=float(
                    rr[
                        "meas_error_N"
                    ]
                    /
                    TARGET_N
                ),

                cmd_pass_5N=bool(
                    rr[
                        "cmd_error_N"
                    ]
                    <=
                    TARGET_N
                ),

                meas_pass_5N=bool(
                    rr[
                        "meas_error_N"
                    ]
                    <=
                    TARGET_N
                ),

                status=(
                    "PASS"
                    if float(
                        rr[
                            "meas_error_N"
                        ]
                    )
                    <= TARGET_N

                    else
                    "FAIL"
                ),
            )

        else:

            # Swing legs were never fed through the loaded-contact
            # FORCE-5A predictor.
            pred_row = dict(

                case_id=int(
                    case_id + 1
                ),

                terrain_type=rec[
                    "terrain_type"
                ],

                time_s=float(
                    d["time"][j]
                ),

                leg=LEG_NAMES_PATCH[
                    leg
                ],

                contact=int(
                    C[
                        leg
                    ] >= 0.5
                ),

                gt_Fx_N=float(
                    F_true[
                        0
                    ]
                ),

                gt_Fy_N=float(
                    F_true[
                        1
                    ]
                ),

                gt_Fz_N=float(
                    F_true[
                        2
                    ]
                ),

                cmd_Fx_N=np.nan,

                cmd_Fy_N=np.nan,

                cmd_Fz_N=np.nan,

                meas_Fx_N=np.nan,

                meas_Fy_N=np.nan,

                meas_Fz_N=np.nan,

                cmd_error_N=np.nan,

                meas_error_N=np.nan,

                meas_error_over_5N=np.nan,

                cmd_pass_5N=False,

                meas_pass_5N=False,

                status=(
                    "SWING — FORCE-5A NOT EVALUATED"
                ),
            )


        prediction_rows.append(
            pred_row
        )


force5_random_sensor_table = pd.DataFrame(
    sensor_rows
)


force5_random_global_table = pd.DataFrame(
    global_rows
)


force5_random_prediction_table = pd.DataFrame(
    prediction_rows
)


# ======================================================================
# 6. PRINT EACH CASE SEPARATELY
# ======================================================================

for case_id in range(
    1,
    5
):

    print()
    print("=" * 120)

    case_info = (
        force5_random_cases
        .iloc[
            case_id - 1
        ]
    )

    print(
        f"TEST CASE {case_id}: "
        f"{case_info['terrain_type']} "
        f"@ t={case_info['time']:.3f} s"
    )

    print("=" * 120)


    print()
    print(
        "BODY STATE + HIGH-LEVEL COMMAND"
    )

    display(
        force5_random_global_table.loc[
            force5_random_global_table[
                "case_id"
            ].eq(
                case_id
            )
        ]
    )


    print()
    print(
        "ALL FOUR LEGS — SENSOR / STATE VALUES"
    )

    display(
        force5_random_sensor_table.loc[
            force5_random_sensor_table[
                "case_id"
            ].eq(
                case_id
            )
        ]
    )


    print()
    print(
        "FORCE RECONSTRUCTION"
    )

    display(
        force5_random_prediction_table.loc[
            force5_random_prediction_table[
                "case_id"
            ].eq(
                case_id
            )
        ]
    )


# ======================================================================
# 7. SAVE EVERYTHING
# ======================================================================

PATCH_ROOT = (
    Path(FORCE5_ROOT)
    /
    "diagnostic_random_cases"
)


PATCH_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


force5_random_cases.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES.csv",
    index=False
)


force5_random_sensor_table.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES_ALL_LEG_SENSORS.csv",
    index=False
)


force5_random_global_table.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES_BODY_COMMAND.csv",
    index=False
)


force5_random_prediction_table.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES_FORCE_RECONSTRUCTION.csv",
    index=False
)


with open(
    PATCH_ROOT /
    "FORCE5A_FIXED_STATUS.json",
    "w"
) as f:

    json.dump(
        dict(
            status=FORCE5A_STATUS,

            cmd_mean_N=float(
                cmd[
                    "mean_vector_error_N"
                ]
            ),

            cmd_p95_N=float(
                cmd[
                    "p95_vector_error_N"
                ]
            ),

            meas_mean_N=float(
                meas[
                    "mean_vector_error_N"
                ]
            ),

            meas_p95_N=float(
                meas[
                    "p95_vector_error_N"
                ]
            ),

            meas_within_5N=float(
                meas[
                    "within_5N_rate"
                ]
            ),
        ),
        f,
        indent=2
    )


print()
print("=" * 120)
print("✅ PATCH COMPLETE")
print("=" * 120)

print(
    "Saved:",
    PATCH_ROOT
)

# %% notebook cell 3
# ======================================================================
# FORCE-5A DIAGNOSTIC PATCH
#
# 1. Fixes KeyError: False
# 2. Finishes FORCE-5A reporting without retraining
# 3. Selects exactly 4 reproducible random test cases:
#       one from each terrain
# 4. Prints ALL available per-leg sensor/state values
# 5. Prints exact force reconstruction results for each stance leg
#
# RUN AFTER THE FAILED FORCE-5A CELL.
# ======================================================================

import numpy as np
import pandas as pd
import json
from pathlib import Path

from IPython.display import display


print("=" * 120)
print("FORCE-5A PATCH + FOUR RANDOM TERRAIN TEST CASES")
print("=" * 120)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_required = [
    "summary",
    "test_raw",
    "force5_meta",
    "run_records",
    "ANYMAL_D_FK",
    "DDQ",
    "TARGET_N",
]

_missing = [
    x for x in _required
    if x not in globals()
]

if _missing:

    raise RuntimeError(
        "Missing FORCE-5A objects: "
        +
        repr(_missing)
        +
        "\nThe large FORCE-5A cell must have reached evaluation first."
    )


# ======================================================================
# 1. FIX THE EXACT PANDAS BUG
#
# WRONG:
#
#     summary.mode
#
# because .mode is a DataFrame method.
#
# RIGHT:
#
#     summary["mode"]
# ======================================================================

cmd_rows = summary.loc[
    summary["mode"].eq(
        "CMD_ORACLE"
    )
]

meas_rows = summary.loc[
    summary["mode"].eq(
        "MEAS_ORACLE"
    )
]


if len(cmd_rows) != 1:

    raise RuntimeError(
        "Expected exactly one CMD_ORACLE summary row."
    )


if len(meas_rows) != 1:

    raise RuntimeError(
        "Expected exactly one MEAS_ORACLE summary row."
    )


cmd = cmd_rows.iloc[0]

meas = meas_rows.iloc[0]


print()
print("✅ Pandas reporting bug fixed.")
print()


print("CMD_ORACLE")

print(
    "  mean       =",
    float(
        cmd["mean_vector_error_N"]
    ),
    "N"
)

print(
    "  p95        =",
    float(
        cmd["p95_vector_error_N"]
    ),
    "N"
)

print(
    "  max        =",
    float(
        cmd["max_vector_error_N"]
    ),
    "N"
)

print(
    "  <=5N rate  =",
    100.0
    *
    float(
        cmd["within_5N_rate"]
    ),
    "%"
)


print()
print("MEAS_ORACLE")

print(
    "  mean       =",
    float(
        meas["mean_vector_error_N"]
    ),
    "N"
)

print(
    "  p95        =",
    float(
        meas["p95_vector_error_N"]
    ),
    "N"
)

print(
    "  max        =",
    float(
        meas["max_vector_error_N"]
    ),
    "N"
)

print(
    "  <=5N rate  =",
    100.0
    *
    float(
        meas["within_5N_rate"]
    ),
    "%"
)


# ======================================================================
# 2. RESTORE A CLEAN RESEARCH STATUS
# ======================================================================

if float(
    meas["p95_vector_error_N"]
) <= TARGET_N:

    FORCE5A_STATUS = (
        "5N_RECONSTRUCTIBLE"
    )

elif float(
    meas["p95_vector_error_N"]
) <= 10.0:

    FORCE5A_STATUS = (
        "NEAR_5N"
    )

else:

    FORCE5A_STATUS = (
        "STATIC_PHYSICS_MODEL_FAILED"
    )


print()
print(
    "FORCE5A_STATUS =",
    FORCE5A_STATUS
)


# ======================================================================
# 3. HELPER — NUMERICAL JACOBIAN
#
# Reuse existing force5_jacobian if alive.
# ======================================================================

if "force5_jacobian" not in globals():

    def force5_jacobian(
        q3,
        leg,
        eps=1e-4,
    ):

        q3 = np.asarray(
            q3,
            np.float64
        )

        q4 = np.zeros(
            (
                4,
                3
            ),
            np.float64
        )

        q4[
            int(leg)
        ] = q3

        p0 = np.asarray(
            ANYMAL_D_FK
            .foot_positions(
                q4
            )[
                int(leg)
            ],
            np.float64
        )

        J = np.zeros(
            (
                3,
                3
            ),
            np.float64
        )

        for k in range(3):

            qq = q4.copy()

            qq[
                int(leg),
                k
            ] += eps

            pp = np.asarray(
                ANYMAL_D_FK
                .foot_positions(
                    qq
                )[
                    int(leg)
                ],
                np.float64
            )

            J[
                :,
                k
            ] = (
                pp - p0
            ) / eps

        return J.astype(
            np.float32
        )


# ======================================================================
# 4. CHOOSE EXACTLY ONE RANDOM STRICT-TEST TIME PER TERRAIN
#
# Fixed seed -> reproducible.
#
# Anchor times come from test_raw, meaning they are guaranteed to lie
# inside the strict test region.
# ======================================================================

RANDOM_TEST_SEED = 20260909

rng = np.random.default_rng(
    RANDOM_TEST_SEED
)


terrain_order = [
    "forest",
    "swamp_marsh",
    "rough_uneven",
    "wet_uneven_proxy",
]


selected_cases = []


for terrain in terrain_order:

    q = test_raw.loc[
        test_raw[
            "terrain_type"
        ].eq(
            terrain
        )
    ].copy()

    if len(q) == 0:

        raise RuntimeError(
            f"No strict test rows for {terrain}"
        )

    # Select among unique physical sample indices,
    # not among individual leg rows.
    candidates = (
        q[
            [
                "ri",
                "index",
                "time",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    r = candidates.iloc[
        int(
            rng.integers(
                0,
                len(candidates)
            )
        )
    ]

    selected_cases.append(
        dict(
            terrain_type=terrain,

            ri=int(
                r["ri"]
            ),

            index=int(
                r["index"]
            ),

            time=float(
                r["time"]
            ),
        )
    )


force5_random_cases = pd.DataFrame(
    selected_cases
)


print()
print("=" * 120)
print("FOUR RANDOM STRICT-TEST CASES")
print("=" * 120)

display(
    force5_random_cases
)


# ======================================================================
# 5. COLLECT EVERY LEGWISE PHYSICS/SENSOR PARAMETER
# ======================================================================

LEG_NAMES_PATCH = [
    "FL",
    "FR",
    "RL",
    "RR",
]


sensor_rows = []

global_rows = []

prediction_rows = []


for case_id, c in force5_random_cases.iterrows():

    ri = int(
        c["ri"]
    )

    j = int(
        c["index"]
    )

    rec = run_records[
        ri
    ]

    d = rec[
        "data"
    ]

    P = np.asarray(
        d["P"][j],
        np.float32
    )

    B = np.asarray(
        d["B"][j],
        np.float32
    )

    U = np.asarray(
        d["U"][j],
        np.float32
    )

    L = np.asarray(
        d["L_LEG"][j],
        np.float32
    )

    C = np.asarray(
        d["contact"][j],
        np.float32
    )

    E = (
        np.asarray(
            d["E"][j],
            np.float32
        )
        if "E" in d
        else
        np.full(
            (
                4,
                3
            ),
            np.nan,
            np.float32
        )
    )

    ddq = np.asarray(
        DDQ[
            ri
        ][
            j
        ],
        np.float32
    )

    q4 = P[
        :,
        0:3
    ]

    dq4 = P[
        :,
        3:6
    ]

    feet = (
        ANYMAL_D_FK
        .foot_positions(
            q4
        )
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------------
    # Global body + command values.
    # --------------------------------------------------------------

    gr = dict(

        case_id=int(
            case_id + 1
        ),

        terrain_type=rec[
            "terrain_type"
        ],

        environment=rec[
            "environment"
        ],

        run=rec[
            "run"
        ],

        index=j,

        time_s=float(
            d["time"][j]
        ),
    )


    for k in range(
        len(B)
    ):

        gr[
            f"B{k}"
        ] = float(
            B[k]
        )


    for k in range(
        len(U)
    ):

        gr[
            f"U{k}"
        ] = float(
            U[k]
        )


    global_rows.append(
        gr
    )


    # --------------------------------------------------------------
    # Per leg.
    # --------------------------------------------------------------

    for leg in range(4):

        J = force5_jacobian(
            q4[
                leg
            ],
            leg
        )

        foot_vel = (
            J
            @
            dq4[
                leg
            ]
        )

        F_true = P[
            leg,
            9:12
        ]

        tau = P[
            leg,
            6:9
        ]


        sr = dict(

            case_id=int(
                case_id + 1
            ),

            terrain_type=rec[
                "terrain_type"
            ],

            time_s=float(
                d["time"][j]
            ),

            leg=LEG_NAMES_PATCH[
                leg
            ],

            contact=int(
                C[
                    leg
                ] >= 0.5
            ),

            # joint position
            q1=float(
                P[
                    leg,
                    0
                ]
            ),

            q2=float(
                P[
                    leg,
                    1
                ]
            ),

            q3=float(
                P[
                    leg,
                    2
                ]
            ),

            # velocity
            dq1=float(
                P[
                    leg,
                    3
                ]
            ),

            dq2=float(
                P[
                    leg,
                    4
                ]
            ),

            dq3=float(
                P[
                    leg,
                    5
                ]
            ),

            # acceleration
            ddq1=float(
                ddq[
                    leg,
                    0
                ]
            ),

            ddq2=float(
                ddq[
                    leg,
                    1
                ]
            ),

            ddq3=float(
                ddq[
                    leg,
                    2
                ]
            ),

            # measured torque
            tau1_Nm=float(
                tau[
                    0
                ]
            ),

            tau2_Nm=float(
                tau[
                    1
                ]
            ),

            tau3_Nm=float(
                tau[
                    2
                ]
            ),

            # low-level commanded torque
            cmd_tau1_Nm=float(
                L[
                    leg,
                    0
                ]
            ),

            cmd_tau2_Nm=float(
                L[
                    leg,
                    1
                ]
            ),

            cmd_tau3_Nm=float(
                L[
                    leg,
                    2
                ]
            ),

            # historical command-response error
            E1_Nm=float(
                E[
                    leg,
                    0
                ]
            ),

            E2_Nm=float(
                E[
                    leg,
                    1
                ]
            ),

            E3_Nm=float(
                E[
                    leg,
                    2
                ]
            ),

            # measured contact force
            Fx_N=float(
                F_true[
                    0
                ]
            ),

            Fy_N=float(
                F_true[
                    1
                ]
            ),

            Fz_N=float(
                F_true[
                    2
                ]
            ),

            Fmag_N=float(
                np.linalg.norm(
                    F_true
                )
            ),

            # FK
            foot_x_m=float(
                feet[
                    leg,
                    0
                ]
            ),

            foot_y_m=float(
                feet[
                    leg,
                    1
                ]
            ),

            foot_z_m=float(
                feet[
                    leg,
                    2
                ]
            ),

            foot_vx_mps=float(
                foot_vel[
                    0
                ]
            ),

            foot_vy_mps=float(
                foot_vel[
                    1
                ]
            ),

            foot_vz_mps=float(
                foot_vel[
                    2
                ]
            ),

            jacobian_condition=float(
                np.linalg.cond(
                    J
                )
            ),
        )


        sensor_rows.append(
            sr
        )


        # ----------------------------------------------------------
        # Prediction result at EXACT SAME physical sample.
        #
        # test_raw contains rows only for loaded-contact legs.
        # ----------------------------------------------------------

        m = test_raw.loc[

            test_raw[
                "ri"
            ].eq(
                ri
            )

            &

            test_raw[
                "index"
            ].eq(
                j
            )

            &

            test_raw[
                "leg"
            ].eq(
                leg
            )
        ]


        if len(m):

            rr = m.iloc[
                0
            ]


            pred_row = dict(

                case_id=int(
                    case_id + 1
                ),

                terrain_type=rec[
                    "terrain_type"
                ],

                time_s=float(
                    d["time"][j]
                ),

                leg=LEG_NAMES_PATCH[
                    leg
                ],

                contact=1,

                gt_Fx_N=float(
                    rr[
                        "gt_Fx_N"
                    ]
                ),

                gt_Fy_N=float(
                    rr[
                        "gt_Fy_N"
                    ]
                ),

                gt_Fz_N=float(
                    rr[
                        "gt_Fz_N"
                    ]
                ),

                cmd_Fx_N=float(
                    rr[
                        "cmd_Fx_N"
                    ]
                ),

                cmd_Fy_N=float(
                    rr[
                        "cmd_Fy_N"
                    ]
                ),

                cmd_Fz_N=float(
                    rr[
                        "cmd_Fz_N"
                    ]
                ),

                meas_Fx_N=float(
                    rr[
                        "meas_Fx_N"
                    ]
                ),

                meas_Fy_N=float(
                    rr[
                        "meas_Fy_N"
                    ]
                ),

                meas_Fz_N=float(
                    rr[
                        "meas_Fz_N"
                    ]
                ),

                cmd_error_N=float(
                    rr[
                        "cmd_error_N"
                    ]
                ),

                meas_error_N=float(
                    rr[
                        "meas_error_N"
                    ]
                ),

                meas_error_over_5N=float(
                    rr[
                        "meas_error_N"
                    ]
                    /
                    TARGET_N
                ),

                cmd_pass_5N=bool(
                    rr[
                        "cmd_error_N"
                    ]
                    <=
                    TARGET_N
                ),

                meas_pass_5N=bool(
                    rr[
                        "meas_error_N"
                    ]
                    <=
                    TARGET_N
                ),

                status=(
                    "PASS"
                    if float(
                        rr[
                            "meas_error_N"
                        ]
                    )
                    <= TARGET_N

                    else
                    "FAIL"
                ),
            )

        else:

            # Swing legs were never fed through the loaded-contact
            # FORCE-5A predictor.
            pred_row = dict(

                case_id=int(
                    case_id + 1
                ),

                terrain_type=rec[
                    "terrain_type"
                ],

                time_s=float(
                    d["time"][j]
                ),

                leg=LEG_NAMES_PATCH[
                    leg
                ],

                contact=int(
                    C[
                        leg
                    ] >= 0.5
                ),

                gt_Fx_N=float(
                    F_true[
                        0
                    ]
                ),

                gt_Fy_N=float(
                    F_true[
                        1
                    ]
                ),

                gt_Fz_N=float(
                    F_true[
                        2
                    ]
                ),

                cmd_Fx_N=np.nan,

                cmd_Fy_N=np.nan,

                cmd_Fz_N=np.nan,

                meas_Fx_N=np.nan,

                meas_Fy_N=np.nan,

                meas_Fz_N=np.nan,

                cmd_error_N=np.nan,

                meas_error_N=np.nan,

                meas_error_over_5N=np.nan,

                cmd_pass_5N=False,

                meas_pass_5N=False,

                status=(
                    "SWING — FORCE-5A NOT EVALUATED"
                ),
            )


        prediction_rows.append(
            pred_row
        )


force5_random_sensor_table = pd.DataFrame(
    sensor_rows
)


force5_random_global_table = pd.DataFrame(
    global_rows
)


force5_random_prediction_table = pd.DataFrame(
    prediction_rows
)


# ======================================================================
# 6. PRINT EACH CASE SEPARATELY
# ======================================================================

for case_id in range(
    1,
    5
):

    print()
    print("=" * 120)

    case_info = (
        force5_random_cases
        .iloc[
            case_id - 1
        ]
    )

    print(
        f"TEST CASE {case_id}: "
        f"{case_info['terrain_type']} "
        f"@ t={case_info['time']:.3f} s"
    )

    print("=" * 120)


    print()
    print(
        "BODY STATE + HIGH-LEVEL COMMAND"
    )

    display(
        force5_random_global_table.loc[
            force5_random_global_table[
                "case_id"
            ].eq(
                case_id
            )
        ]
    )


    print()
    print(
        "ALL FOUR LEGS — SENSOR / STATE VALUES"
    )

    display(
        force5_random_sensor_table.loc[
            force5_random_sensor_table[
                "case_id"
            ].eq(
                case_id
            )
        ]
    )


    print()
    print(
        "FORCE RECONSTRUCTION"
    )

    display(
        force5_random_prediction_table.loc[
            force5_random_prediction_table[
                "case_id"
            ].eq(
                case_id
            )
        ]
    )


# ======================================================================
# 7. SAVE EVERYTHING
# ======================================================================

PATCH_ROOT = (
    Path(FORCE5_ROOT)
    /
    "diagnostic_random_cases"
)


PATCH_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


force5_random_cases.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES.csv",
    index=False
)


force5_random_sensor_table.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES_ALL_LEG_SENSORS.csv",
    index=False
)


force5_random_global_table.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES_BODY_COMMAND.csv",
    index=False
)


force5_random_prediction_table.to_csv(
    PATCH_ROOT /
    "FOUR_RANDOM_CASES_FORCE_RECONSTRUCTION.csv",
    index=False
)


with open(
    PATCH_ROOT /
    "FORCE5A_FIXED_STATUS.json",
    "w"
) as f:

    json.dump(
        dict(
            status=FORCE5A_STATUS,

            cmd_mean_N=float(
                cmd[
                    "mean_vector_error_N"
                ]
            ),

            cmd_p95_N=float(
                cmd[
                    "p95_vector_error_N"
                ]
            ),

            meas_mean_N=float(
                meas[
                    "mean_vector_error_N"
                ]
            ),

            meas_p95_N=float(
                meas[
                    "p95_vector_error_N"
                ]
            ),

            meas_within_5N=float(
                meas[
                    "within_5N_rate"
                ]
            ),
        ),
        f,
        indent=2
    )


print()
print("=" * 120)
print("✅ PATCH COMPLETE")
print("=" * 120)

print(
    "Saved:",
    PATCH_ROOT
)

# %% notebook cell 5
# ======================================================================
# FORCE-5A VISUAL DIAGNOSTIC SUITE
#
# Produces:
#
# A. Four-terrain legwise error/risk maps
# B. Four selected random-case risk maps
# C. Hind-leg RL / RR force reconstruction over time
# D. Per-leg 3D force clouds:
#       true vs CMD vs MEAS
#       centroid
#       mean radius
#       median radius
#       p95 radius
# E. Error distributions by terrain/leg
#
# IMPORTANT:
# "risk" here = force-prediction diagnostic risk.
# It is NOT yet a terrain foothold planner risk.
# ======================================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path


print("=" * 120)
print("FORCE-5A VISUAL DIAGNOSTICS")
print("=" * 120)


_required = [
    "test_raw",
    "force5_random_prediction_table",
    "force5_random_sensor_table",
    "TARGET_N",
    "FORCE5_ROOT",
]

_missing = [
    x for x in _required
    if x not in globals()
]

if _missing:

    raise RuntimeError(
        "Run the diagnostic patch cell first. Missing: "
        +
        repr(_missing)
    )


VIS_ROOT = (
    Path(FORCE5_ROOT)
    /
    "diagnostic_visualizations"
)

VIS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


LEG_ORDER = [
    "FL",
    "FR",
    "RL",
    "RR",
]


TERRAIN_ORDER = [
    "forest",
    "swamp_marsh",
    "rough_uneven",
    "wet_uneven_proxy",
]


# ======================================================================
# 1. TEST-WIDE TERRAIN × LEG RISK TABLE
#
# risk ratio:
#
#       error / 5 N
#
# 1.0 = user's maximum allowable threshold.
# ======================================================================

risk_rows = []


for (
    terrain,
    leg
), g in test_raw.groupby(
    [
        "terrain_type",
        "leg_name",
    ]
):

    risk_rows.append(
        dict(
            terrain_type=terrain,

            leg_name=leg,

            samples=int(
                len(g)
            ),

            cmd_mean_error_N=float(
                g[
                    "cmd_error_N"
                ].mean()
            ),

            meas_mean_error_N=float(
                g[
                    "meas_error_N"
                ].mean()
            ),

            cmd_p95_error_N=float(
                g[
                    "cmd_error_N"
                ].quantile(
                    0.95
                )
            ),

            meas_p95_error_N=float(
                g[
                    "meas_error_N"
                ].quantile(
                    0.95
                )
            ),

            cmd_risk_ratio=float(
                g[
                    "cmd_error_N"
                ].mean()
                /
                TARGET_N
            ),

            meas_risk_ratio=float(
                g[
                    "meas_error_N"
                ].mean()
                /
                TARGET_N
            ),

            meas_5N_success_rate=float(
                np.mean(
                    g[
                        "meas_error_N"
                    ]
                    <=
                    TARGET_N
                )
            ),
        )
    )


force5_risk_table = pd.DataFrame(
    risk_rows
)


print()
print(
    "TEST-WIDE TERRAIN × LEG RISK"
)

display(
    force5_risk_table
)


# ======================================================================
# 2. RISK MAP — MEAN ERROR / 5 N
# ======================================================================

risk_pivot = (
    force5_risk_table
    .pivot(
        index="terrain_type",
        columns="leg_name",
        values="meas_risk_ratio",
    )
    .reindex(
        index=TERRAIN_ORDER,
        columns=LEG_ORDER,
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        5
    )
)


im = ax.imshow(
    risk_pivot.values,
    aspect="auto"
)


ax.set_xticks(
    np.arange(
        len(
            LEG_ORDER
        )
    )
)

ax.set_xticklabels(
    LEG_ORDER
)


ax.set_yticks(
    np.arange(
        len(
            TERRAIN_ORDER
        )
    )
)

ax.set_yticklabels(
    TERRAIN_ORDER
)


ax.set_title(
    "MEAS_ORACLE prediction risk = mean 3D error / 5 N"
)

ax.set_xlabel(
    "leg"
)

ax.set_ylabel(
    "terrain"
)


for i in range(
    risk_pivot.shape[
        0
    ]
):

    for j in range(
        risk_pivot.shape[
            1
        ]
    ):

        v = risk_pivot.values[
            i,
            j
        ]

        ax.text(
            j,
            i,
            f"{v:.2f}×",
            ha="center",
            va="center",
        )


fig.colorbar(
    im,
    ax=ax,
    label="error / 5 N"
)


fig.tight_layout()


fig.savefig(
    VIS_ROOT /
    "RISK_MAP_TERRAIN_LEG_MEAN.png",
    dpi=180,
    bbox_inches="tight"
)


plt.show()


# ======================================================================
# 3. P95 RISK MAP
# ======================================================================

p95_pivot = (
    force5_risk_table
    .assign(
        p95_risk=lambda x:
        x[
            "meas_p95_error_N"
        ]
        /
        TARGET_N
    )
    .pivot(
        index="terrain_type",
        columns="leg_name",
        values="p95_risk",
    )
    .reindex(
        index=TERRAIN_ORDER,
        columns=LEG_ORDER,
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        5
    )
)


im = ax.imshow(
    p95_pivot.values,
    aspect="auto"
)


ax.set_xticks(
    np.arange(
        4
    )
)

ax.set_xticklabels(
    LEG_ORDER
)


ax.set_yticks(
    np.arange(
        4
    )
)

ax.set_yticklabels(
    TERRAIN_ORDER
)


ax.set_title(
    "p95 force-error risk relative to 5 N requirement"
)


for i in range(
    4
):

    for j in range(
        4
    ):

        ax.text(
            j,
            i,
            f"{p95_pivot.values[i,j]:.1f}×",
            ha="center",
            va="center",
        )


fig.colorbar(
    im,
    ax=ax,
    label="p95 error / 5 N"
)


fig.tight_layout()


fig.savefig(
    VIS_ROOT /
    "RISK_MAP_TERRAIN_LEG_P95.png",
    dpi=180,
    bbox_inches="tight"
)


plt.show()


# ======================================================================
# 4. FOUR RANDOM CASE RISK MAP
# ======================================================================

random_risk = (
    force5_random_prediction_table
    .copy()
)


random_risk[
    "risk"
] = (
    random_risk[
        "meas_error_N"
    ]
    /
    TARGET_N
)


case_risk_pivot = (
    random_risk
    .pivot(
        index="case_id",
        columns="leg",
        values="risk",
    )
    .reindex(
        index=[
            1,
            2,
            3,
            4,
        ],
        columns=LEG_ORDER,
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        4
    )
)


masked = np.ma.masked_invalid(
    case_risk_pivot.values
)


im = ax.imshow(
    masked,
    aspect="auto"
)


ax.set_xticks(
    np.arange(
        4
    )
)

ax.set_xticklabels(
    LEG_ORDER
)


ax.set_yticks(
    np.arange(
        4
    )
)

ax.set_yticklabels(
    [
        "case 1",
        "case 2",
        "case 3",
        "case 4",
    ]
)


ax.set_title(
    "Four random cases — stance-leg force-error risk"
)


for i in range(
    4
):

    for j in range(
        4
    ):

        v = case_risk_pivot.values[
            i,
            j
        ]

        if np.isfinite(v):

            txt = f"{v:.2f}×"

        else:

            txt = "swing"

        ax.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
        )


fig.colorbar(
    im,
    ax=ax,
    label="MEAS error / 5 N"
)


fig.tight_layout()


fig.savefig(
    VIS_ROOT /
    "RISK_MAP_FOUR_RANDOM_CASES.png",
    dpi=180,
    bbox_inches="tight"
)


plt.show()


# ======================================================================
# 5. HIND LEG RECONSTRUCTION
#
# RL + RR.
#
# For readability:
# plot force magnitude and reconstruction error versus time.
# ======================================================================

hind = test_raw.loc[
    test_raw[
        "leg_name"
    ].isin(
        [
            "RL",
            "RR",
        ]
    )
].copy()


hind[
    "gt_mag_N"
] = np.sqrt(

    hind[
        "gt_Fx_N"
    ] ** 2

    +

    hind[
        "gt_Fy_N"
    ] ** 2

    +

    hind[
        "gt_Fz_N"
    ] ** 2
)


hind[
    "cmd_mag_N"
] = np.sqrt(

    hind[
        "cmd_Fx_N"
    ] ** 2

    +

    hind[
        "cmd_Fy_N"
    ] ** 2

    +

    hind[
        "cmd_Fz_N"
    ] ** 2
)


hind[
    "meas_mag_N"
] = np.sqrt(

    hind[
        "meas_Fx_N"
    ] ** 2

    +

    hind[
        "meas_Fy_N"
    ] ** 2

    +

    hind[
        "meas_Fz_N"
    ] ** 2
)


for terrain in TERRAIN_ORDER:

    for leg in [
        "RL",
        "RR",
    ]:

        g = (
            hind.loc[
                hind[
                    "terrain_type"
                ].eq(
                    terrain
                )

                &

                hind[
                    "leg_name"
                ].eq(
                    leg
                )
            ]
            .sort_values(
                "time"
            )
        )


        if len(g) == 0:

            continue


        fig, ax = plt.subplots(
            figsize=(
                11,
                4
            )
        )


        ax.plot(
            g[
                "time"
            ],
            g[
                "gt_mag_N"
            ],
            label="ground truth |F|",
            linewidth=1.5,
        )


        ax.plot(
            g[
                "time"
            ],
            g[
                "cmd_mag_N"
            ],
            label="CMD oracle |F|",
            linewidth=1.0,
        )


        ax.plot(
            g[
                "time"
            ],
            g[
                "meas_mag_N"
            ],
            label="MEAS oracle |F|",
            linewidth=1.0,
        )


        ax.set_title(
            f"{terrain} — {leg} hind-leg force reconstruction"
        )

        ax.set_xlabel(
            "time [s]"
        )

        ax.set_ylabel(
            "force magnitude [N]"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()


        fig.savefig(
            VIS_ROOT /
            f"HIND_RECON_{terrain}_{leg}.png",
            dpi=180,
            bbox_inches="tight"
        )


        plt.show()


# ======================================================================
# 6. HIND-LEG ERROR THROUGH TIME
# ======================================================================

for terrain in TERRAIN_ORDER:

    g = (
        hind.loc[
            hind[
                "terrain_type"
            ].eq(
                terrain
            )
        ]
        .sort_values(
            "time"
        )
    )


    if len(g) == 0:

        continue


    fig, ax = plt.subplots(
        figsize=(
            11,
            4
        )
    )


    for leg in [
        "RL",
        "RR",
    ]:

        q = g.loc[
            g[
                "leg_name"
            ].eq(
                leg
            )
        ]

        ax.plot(
            q[
                "time"
            ],
            q[
                "meas_error_N"
            ],
            label=leg,
            linewidth=1.0,
        )


    ax.axhline(
        TARGET_N,
        linestyle="--",
        label="5 N requirement",
    )


    ax.set_title(
        f"{terrain} — hind-leg 3D force-vector error"
    )

    ax.set_xlabel(
        "time [s]"
    )

    ax.set_ylabel(
        "3D vector error [N]"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()


    fig.savefig(
        VIS_ROOT /
        f"HIND_ERROR_{terrain}.png",
        dpi=180,
        bbox_inches="tight"
    )


    plt.show()


# ======================================================================
# 7. FORCE CLOUD STATISTICS
#
# Cloud = distribution in physical Fx/Fy/Fz space.
# ======================================================================

cloud_rows = []


def cloud_stats(
    xyz
):

    xyz = np.asarray(
        xyz,
        np.float64
    )

    centroid = xyz.mean(
        axis=0
    )

    radius = np.linalg.norm(
        xyz
        -
        centroid[
            None,
            :
        ],
        axis=1
    )


    return dict(

        centroid_Fx_N=float(
            centroid[
                0
            ]
        ),

        centroid_Fy_N=float(
            centroid[
                1
            ]
        ),

        centroid_Fz_N=float(
            centroid[
                2
            ]
        ),

        mean_radius_N=float(
            radius.mean()
        ),

        median_radius_N=float(
            np.median(
                radius
            )
        ),

        p95_radius_N=float(
            np.quantile(
                radius,
                0.95
            )
        ),

        max_radius_N=float(
            radius.max()
        ),
    )


for leg in LEG_ORDER:

    g = test_raw.loc[
        test_raw[
            "leg_name"
        ].eq(
            leg
        )
    ]


    gt_xyz = g[
        [
            "gt_Fx_N",
            "gt_Fy_N",
            "gt_Fz_N",
        ]
    ].to_numpy()


    cmd_xyz = g[
        [
            "cmd_Fx_N",
            "cmd_Fy_N",
            "cmd_Fz_N",
        ]
    ].to_numpy()


    meas_xyz = g[
        [
            "meas_Fx_N",
            "meas_Fy_N",
            "meas_Fz_N",
        ]
    ].to_numpy()


    for name, xyz in [
        (
            "GROUND_TRUTH",
            gt_xyz
        ),
        (
            "CMD_ORACLE",
            cmd_xyz
        ),
        (
            "MEAS_ORACLE",
            meas_xyz
        ),
    ]:

        cloud_rows.append(
            dict(
                leg=leg,

                cloud=name,

                samples=len(
                    xyz
                ),

                **cloud_stats(
                    xyz
                ),
            )
        )


force5_force_cloud_stats = pd.DataFrame(
    cloud_rows
)


print()
print("=" * 120)
print("LEGWISE FORCE-CLOUD STATISTICS")
print("=" * 120)

display(
    force5_force_cloud_stats
)


# ======================================================================
# 8. LEGWISE 3D FORCE CLOUDS
#
# Separate plot per leg for clarity.
# ======================================================================

for leg in LEG_ORDER:

    g = test_raw.loc[
        test_raw[
            "leg_name"
        ].eq(
            leg
        )
    ]


    gt_xyz = g[
        [
            "gt_Fx_N",
            "gt_Fy_N",
            "gt_Fz_N",
        ]
    ].to_numpy()


    meas_xyz = g[
        [
            "meas_Fx_N",
            "meas_Fy_N",
            "meas_Fz_N",
        ]
    ].to_numpy()


    # Keep plot readable while preserving full statistics.
    rng = np.random.default_rng(
        781
    )

    nplot = min(
        1500,
        len(
            g
        )
    )


    ids = rng.choice(
        len(
            g
        ),
        size=nplot,
        replace=False,
    )


    fig = plt.figure(
        figsize=(
            8,
            7
        )
    )


    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(
        gt_xyz[
            ids,
            0
        ],
        gt_xyz[
            ids,
            1
        ],
        gt_xyz[
            ids,
            2
        ],
        s=8,
        alpha=0.30,
        label="ground truth",
    )


    ax.scatter(
        meas_xyz[
            ids,
            0
        ],
        meas_xyz[
            ids,
            1
        ],
        meas_xyz[
            ids,
            2
        ],
        s=8,
        alpha=0.30,
        label="MEAS reconstruction",
    )


    gt_centroid = gt_xyz.mean(
        axis=0
    )

    meas_centroid = meas_xyz.mean(
        axis=0
    )


    ax.scatter(
        [
            gt_centroid[
                0
            ]
        ],
        [
            gt_centroid[
                1
            ]
        ],
        [
            gt_centroid[
                2
            ]
        ],
        s=110,
        marker="x",
        label="GT centroid",
    )


    ax.scatter(
        [
            meas_centroid[
                0
            ]
        ],
        [
            meas_centroid[
                1
            ]
        ],
        [
            meas_centroid[
                2
            ]
        ],
        s=110,
        marker="x",
        label="pred centroid",
    )


    row = force5_force_cloud_stats.loc[

        force5_force_cloud_stats[
            "leg"
        ].eq(
            leg
        )

        &

        force5_force_cloud_stats[
            "cloud"
        ].eq(
            "GROUND_TRUTH"
        )
    ].iloc[0]


    ax.set_title(

        f"{leg} force cloud\n"
        f"GT mean radius={row['mean_radius_N']:.1f} N, "
        f"p95 radius={row['p95_radius_N']:.1f} N"
    )


    ax.set_xlabel(
        "Fx [N]"
    )

    ax.set_ylabel(
        "Fy [N]"
    )

    ax.set_zlabel(
        "Fz [N]"
    )


    ax.legend()


    fig.tight_layout()


    fig.savefig(
        VIS_ROOT /
        f"FORCE_CLOUD_{leg}.png",
        dpi=180,
        bbox_inches="tight"
    )


    plt.show()


# ======================================================================
# 9. ERROR CLOUD / DISTRIBUTION TABLE
# ======================================================================

error_rows = []


for (
    terrain,
    leg
), g in test_raw.groupby(
    [
        "terrain_type",
        "leg_name",
    ]
):

    e = g[
        "meas_error_N"
    ].to_numpy(
        np.float64
    )


    error_rows.append(
        dict(
            terrain_type=terrain,

            leg_name=leg,

            n=len(e),

            mean_N=float(
                e.mean()
            ),

            median_N=float(
                np.median(
                    e
                )
            ),

            std_N=float(
                e.std()
            ),

            p90_N=float(
                np.quantile(
                    e,
                    0.90
                )
            ),

            p95_N=float(
                np.quantile(
                    e,
                    0.95
                )
            ),

            p99_N=float(
                np.quantile(
                    e,
                    0.99
                )
            ),

            within_5N=float(
                np.mean(
                    e
                    <=
                    TARGET_N
                )
            ),
        )
    )


force5_error_cloud_stats = pd.DataFrame(
    error_rows
)


print()
print("=" * 120)
print("ERROR-DISTRIBUTION STATISTICS")
print("=" * 120)

display(
    force5_error_cloud_stats
)


# ======================================================================
# 10. SAVE TABLES
# ======================================================================

force5_risk_table.to_csv(
    VIS_ROOT /
    "RISK_TABLE_TERRAIN_LEG.csv",
    index=False
)


force5_force_cloud_stats.to_csv(
    VIS_ROOT /
    "LEGWISE_FORCE_CLOUD_STATS.csv",
    index=False
)


force5_error_cloud_stats.to_csv(
    VIS_ROOT /
    "LEGWISE_ERROR_DISTRIBUTIONS.csv",
    index=False
)


hind.to_csv(
    VIS_ROOT /
    "HIND_LEG_RECONSTRUCTION_RAW.csv",
    index=False
)


print()
print("=" * 120)
print("✅ VISUAL DIAGNOSTIC SUITE COMPLETE")
print("=" * 120)

print(
    "Saved:",
    VIS_ROOT
)

# %% notebook cell 7
# ======================================================================
# V10-0 — RAW MULTIMODAL RESTORE + EXTRINSIC SELF-CALIBRATION
# Run after the existing FORCE-5 bootstrap cell in this notebook.
# ======================================================================
import os, sys, subprocess, json, math, time, gc, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from IPython.display import display

_req=['run_records','ANYMAL_D_FK','PERSIST_ROOT','CACHE_ROOT','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing:
    raise RuntimeError('Run the FORCE-5 fresh-runtime bootstrap first. Missing: '+repr(_missing))

V10_ROOT=Path(PERSIST_ROOT)/'V10_geometry_coupled_force'
V10_RAW_ROOT=V10_ROOT/'raw'
V10_GEOM_ROOT=V10_ROOT/'geometry_memory'
V10_DATA_ROOT=V10_ROOT/'datasets'
V10_MODEL_ROOT=V10_ROOT/'models'
V10_EVAL_ROOT=V10_ROOT/'evaluation'
V10_PLAN_ROOT=V10_ROOT/'planner'
for p in [V10_ROOT,V10_RAW_ROOT,V10_GEOM_ROOT,V10_DATA_ROOT,V10_MODEL_ROOT,V10_EVAL_ROOT,V10_PLAN_ROOT]:
    p.mkdir(parents=True,exist_ok=True)

V10_SEED=108781
random.seed(V10_SEED); np.random.seed(V10_SEED); torch.manual_seed(V10_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(V10_SEED)

# Camera model already used by V7/TartanGround front camera.
CAM_W=640; CAM_H=640; FX=320.0; FY=320.0; CX=320.0; CY=320.0
RAW_GRID=8
V10_RAW_DATA_ROOT=Path('/content/tartanground_v10')
V10_RAW_DATA_ROOT.mkdir(parents=True,exist_ok=True)
V10_TEACHER_CAMERAS=['lcam_front','lcam_back','lcam_left','lcam_right','lcam_top','lcam_bottom']
V10_USE_ALL_CAMERAS=True

# ----------------------------------------------------------------------
# Install only the raw-data packages not guaranteed by the bootstrap.
# ----------------------------------------------------------------------
def _ensure_import(import_name,pip_spec):
    try:
        return __import__(import_name)
    except Exception:
        print('Installing',pip_spec)
        subprocess.check_call([sys.executable,'-m','pip','install','-q',pip_spec])
        return __import__(import_name)

try:
    import tartanair as ta
except Exception:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','git+https://github.com/castacks/tartanairpy.git'])
    import tartanair as ta

# ----------------------------------------------------------------------
# Mandatory raw RGB/depth/meta restore. ROS-bag signals are already in the
# persistent physics caches, but raw bag is restored too for audit/replay.
# ----------------------------------------------------------------------
def v10_restore_raw_run(rec):
    env,run=rec['environment'],rec['run']
    base=V10_RAW_DATA_ROOT/env/'Data_anymal'/run
    required=[base/'imu',base/'rosbags'/f'{run}.bag',base/'image_lcam_front',base/'depth_lcam_front',base/'pose_lcam_front.txt']
    if not all(p.exists() for p in required):
        ta.init(str(V10_RAW_DATA_ROOT))
        print('Downloading mandatory TartanGround raw modalities:',env,run)
        ta.download_ground(env=[env],version=['anymal'],traj=[run],
                           modality=['imu','rosbag','image','depth','meta'],
                           camera_name=['lcam_front'],unzip=True)
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise RuntimeError('Raw TartanGround restore incomplete: '+repr(missing))
    if V10_USE_ALL_CAMERAS:
        teacher_required=[]
        for cam in V10_TEACHER_CAMERAS:
            teacher_required += [base/f'image_{cam}',base/f'depth_{cam}',base/f'pose_{cam}.txt']
        if not all(p.exists() for p in teacher_required):
            print('Downloading six-camera RGB-D privileged geometry teacher:',env,run)
            ta.init(str(V10_RAW_DATA_ROOT))
            ta.download_ground(env=[env],version=['anymal'],traj=[run],modality=['image','depth','meta'],camera_name=V10_TEACHER_CAMERAS,unzip=True)
    rec['base']=base
    return base

for rec in run_records:
    v10_restore_raw_run(rec)

# Optional privileged teacher modalities. The force model does NOT depend on
# these succeeding. We attempt them once and record what is actually present.
OPTIONAL_MODALITY_STATUS=[]
for rec in run_records:
    env,run=rec['environment'],rec['run']
    for modality in ['seg_labels','sem_pcd']:
        ok=True; msg=''
        try:
            ta.init(str(V10_RAW_DATA_ROOT))
            ta.download_ground(env=[env],version=['anymal'],traj=[run],modality=[modality],
                               camera_name=['lcam_front'],unzip=True)
        except Exception as e:
            ok=False; msg=repr(e)[:240]
        OPTIONAL_MODALITY_STATUS.append(dict(terrain=rec['terrain_type'],modality=modality,download_call_ok=ok,message=msg))
optional_modalities=pd.DataFrame(OPTIONAL_MODALITY_STATUS)
display(optional_modalities)

# ----------------------------------------------------------------------
# Restore / construct a basic causal frame manifest if the bootstrap did not
# attach one. When a cached original manifest exists, it is always preferred.
# ----------------------------------------------------------------------
def _npz_dict(path):
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in z.files}

def v10_restore_manifest(rec):
    if 'manifest' in rec and all(k in rec['manifest'] for k in ['frame_ids','camera_times','poses','visual_index']):
        return rec['manifest']
    env,run=rec['environment'],rec['run']
    cp=Path(CACHE_ROOT)/f'{env}_{run}_frame_manifest.npz'
    if cp.exists():
        rec['manifest']=_npz_dict(cp); return rec['manifest']
    base=rec['base']; data=rec['data']
    rgb_dir=base/'image_lcam_front'; dep_dir=base/'depth_lcam_front'
    rgb_ids={f.name.split('_')[0] for f in rgb_dir.glob('*_lcam_front.png')}
    dep_ids={f.name.split('_')[0] for f in dep_dir.glob('*_lcam_front_depth.png')}
    ids=sorted(rgb_ids&dep_ids,key=lambda x:int(x))
    nums=np.asarray([int(x) for x in ids],np.int64)
    cam_orig=np.loadtxt(base/'imu'/'cam_time.txt').reshape(-1).astype(np.float64)
    poses=np.loadtxt(base/'pose_lcam_front.txt').astype(np.float64)
    times=cam_orig[nums]-float(data['crop_start_original'])
    keep=(times>=data['time'][0]-1e-8)&(times<=data['time'][-1]+1e-8)
    ids=np.asarray(ids)[keep]; nums=nums[keep]; times=times[keep]; poses=poses[nums]
    vi=np.searchsorted(times,data['time'],side='right')-1
    boxes=np.tile(np.array([[0,0,CAM_W,CAM_H]],np.int32),(len(ids),1))
    out=dict(frame_ids=ids.astype('<U16'),frame_nums=nums,camera_times=times,
             poses=poses.astype(np.float64),crop_boxes=boxes,visual_index=vi.astype(np.int64),
             ground_fraction=np.ones(len(ids),np.float32))
    np.savez_compressed(cp,**out); rec['manifest']=out
    return out

for rec in run_records: v10_restore_manifest(rec)

# Attach cached visual branches if bootstrap found them but did not put them in rec.
def v10_attach_visual(rec):
    if 'visual' in rec and all(k in rec['visual'] for k in ['DINO','CONV','FLOW']): return
    env,run=rec['environment'],rec['run']
    paths={
        'DINO':Path(CACHE_ROOT)/f'{env}_{run}_v7_dinov3.npy',
        'CONV':Path(CACHE_ROOT)/f'{env}_{run}_v7_convnext.npy',
        'GEOM':Path(CACHE_ROOT)/f'{env}_{run}_v7_utonia.npy',
        'FLOW':Path(CACHE_ROOT)/f'{env}_{run}_v7_waft.npy',
    }
    if all(p.exists() for p in paths.values()):
        rec['visual']={k:np.load(p,mmap_mode='r') for k,p in paths.items()}
        rec['visual'].update(rec['manifest'])
for rec in run_records: v10_attach_visual(rec)

# ----------------------------------------------------------------------
# Metric depth + pose helpers. Tartan camera axes: x forward,y right,z down;
# pose7 = tx ty tz qx qy qz qw, world NED.
# ----------------------------------------------------------------------
import cv2

def v10_read_depth(path):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None: raise RuntimeError('Could not read depth: '+str(path))
    if raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[2]==4:
        raw=np.ascontiguousarray(raw); d=raw.view('<f4').reshape(raw.shape[0],raw.shape[1])
    elif raw.dtype==np.float32 and raw.ndim==2:
        d=raw
    else: raise RuntimeError(f'Unexpected depth format {raw.dtype} {raw.shape}')
    d=d.astype(np.float32); d[~np.isfinite(d)]=0
    return d

def v10_pose_Rwc(pose7): return Rotation.from_quat(np.asarray(pose7[3:7],np.float64)).as_matrix()

def v10_heading_Rlw(pose7):
    Rwc=v10_pose_Rwc(pose7); f=Rwc[:,0].copy(); fh=np.array([f[0],f[1],0.0],np.float64)
    if np.linalg.norm(fh)<1e-8: fh=np.array([1.,0.,0.])
    fh/=np.linalg.norm(fh); rh=np.array([-fh[1],fh[0],0.0]); up=np.array([0.,0.,-1.])
    return np.stack([fh,rh,up],axis=0)

def v10_pose_at(rec,idx):
    vi=int(rec['manifest']['visual_index'][int(idx)])
    if vi<0: return None,None
    return np.asarray(rec['manifest']['poses'][vi],np.float64),vi

def v10_pose_interp_time(rec,tq):
    # Privileged GT localization used to build training labels / slip diagnostics.
    # Causal perception queries still use v10_pose_at (hold-last-frame).
    tm=np.asarray(rec['manifest']['camera_times'],np.float64); poses=np.asarray(rec['manifest']['poses'],np.float64); tq=float(tq)
    if tq<=tm[0]: return poses[0].copy()
    if tq>=tm[-1]: return poses[-1].copy()
    j=int(np.searchsorted(tm,tq)); i=j-1; a=(tq-tm[i])/max(tm[j]-tm[i],1e-9)
    pos=(1-a)*poses[i,:3]+a*poses[j,:3]
    r=Slerp([tm[i],tm[j]],Rotation.from_quat(np.stack([poses[i,3:7],poses[j,3:7]])))([tq]).as_quat()[0]
    return np.concatenate([pos,r]).astype(np.float64)

def v10_world_from_heading(local_xyz,pose7):
    local=np.asarray(local_xyz,np.float64); Rlw=v10_heading_Rlw(pose7)
    return np.asarray(pose7[:3],np.float64)+local@Rlw

def v10_heading_from_world(world_xyz,pose7):
    world=np.asarray(world_xyz,np.float64); Rlw=v10_heading_Rlw(pose7)
    return (world-np.asarray(pose7[:3],np.float64))@Rlw.T

# ----------------------------------------------------------------------
# Shared camera<-base calibration from stance-foot stationarity.
# Base axes x forward,y left,z up. Camera axes x forward,y right,z down.
# Initial rotation is therefore Rx(pi); translation is solved from data.
# ----------------------------------------------------------------------
R_CB0=np.diag([1.,-1.,-1.])

def _Tcb_from_x(x):
    Rd=Rotation.from_rotvec(np.asarray(x[:3],np.float64)).as_matrix()
    return Rd@R_CB0,np.asarray(x[3:6],np.float64)

def _collect_calibration_pairs(max_pairs=3000):
    pairs=[]
    for ri,rec in enumerate(run_records):
        man=rec['manifest']; data=rec['data']; ct=np.asarray(man['camera_times'],np.float64)
        pidx=np.searchsorted(data['time'],ct,side='left'); pidx=np.clip(pidx,0,len(data['time'])-1)
        for k in range(1,len(pidx)):
            i0,i1=int(pidx[k-1]),int(pidx[k])
            for leg in range(4):
                if data['contact'][i0,leg]>.5 and data['contact'][i1,leg]>.5:
                    pB0=ANYMAL_D_FK.foot_positions(data['P'][i0,:,0:3])[leg].astype(np.float64)
                    pB1=ANYMAL_D_FK.foot_positions(data['P'][i1,:,0:3])[leg].astype(np.float64)
                    pairs.append((np.asarray(man['poses'][k-1],np.float64),np.asarray(man['poses'][k],np.float64),pB0,pB1))
    if len(pairs)>max_pairs:
        rng=np.random.default_rng(V10_SEED); ids=rng.choice(len(pairs),max_pairs,replace=False); pairs=[pairs[i] for i in ids]
    return pairs

V10_CALIB_PATH=V10_ROOT/'camera_base_self_calibration.json'
if V10_CALIB_PATH.exists():
    cal=json.loads(V10_CALIB_PATH.read_text()); V10_TCB_X=np.asarray(cal['x'],np.float64)
else:
    pairs=_collect_calibration_pairs()
    if len(pairs)<100: raise RuntimeError('Too few stance pairs for camera/base self-calibration.')
    def residual(x):
        Rcb,tcb=_Tcb_from_x(x); out=[]
        for p0,p1,b0,b1 in pairs:
            w0=v10_pose_Rwc(p0)@(Rcb@b0+tcb)+p0[:3]
            w1=v10_pose_Rwc(p1)@(Rcb@b1+tcb)+p1[:3]
            out.extend((w1-w0).tolist())
        # weak calibration priors; no guessed translation is forced.
        out.extend((0.03*np.asarray(x[:3])).tolist()); out.extend((0.01*np.asarray(x[3:6])).tolist())
        return np.asarray(out,np.float64)
    sol=least_squares(residual,np.zeros(6),loss='soft_l1',f_scale=.03,
                      bounds=(np.array([-.6,-.6,-.6,-1.5,-1.5,-1.5]),np.array([.6,.6,.6,1.5,1.5,1.5])),max_nfev=250)
    V10_TCB_X=sol.x.astype(np.float64)
    before=np.median(np.linalg.norm(residual(np.zeros(6)).reshape(-1,3),axis=1))
    after=np.median(np.linalg.norm(residual(V10_TCB_X).reshape(-1,3),axis=1))
    V10_CALIB_PATH.write_text(json.dumps(dict(x=V10_TCB_X.tolist(),cost=float(sol.cost),pairs=len(pairs),median_proxy_before=float(before),median_proxy_after=float(after)),indent=2))

V10_R_CB,V10_t_CB=_Tcb_from_x(V10_TCB_X)
print('V10 camera<-base calibration rotvec correction:',V10_TCB_X[:3])
print('V10 camera<-base translation [m]:',V10_t_CB)


def v10_foot_world(rec,idx):
    pose,_=v10_pose_at(rec,idx)
    if pose is None: return None
    pB=ANYMAL_D_FK.foot_positions(rec['data']['P'][int(idx),:,0:3]).astype(np.float64)
    pC=pB@V10_R_CB.T+V10_t_CB[None,:]
    return (pC@v10_pose_Rwc(pose).T+pose[:3][None,:]).astype(np.float32)

# Precompute/cache all base/world foot trajectories once. This avoids hundreds of
# thousands of repeated FK calls in later geometry/contact stages.
V10_FOOT_BASE={}; V10_FOOT_WORLD={}; V10_FOOT_VEL_WORLD={}
for ri,rec in enumerate(run_records):
    env,run=rec['environment'],rec['run']; cp=V10_DATA_ROOT/f'{env}_{run}_FOOT_KINEMATICS.npz'
    if cp.exists():
        z=np.load(cp,allow_pickle=False); fb=z['foot_base']; fw=z['foot_world']; fv=z['foot_vel_world']
    else:
        d=rec['data']; N=len(d['time']); fb=np.empty((N,4,3),np.float32); fw=np.empty((N,4,3),np.float32)
        for j in range(N):
            fb[j]=ANYMAL_D_FK.foot_positions(d['P'][j,:,0:3]).astype(np.float32)
            pose=v10_pose_interp_time(rec,float(d['time'][j]))
            if pose is None: fw[j]=np.nan; continue
            pC=fb[j].astype(np.float64)@V10_R_CB.T+V10_t_CB[None,:]
            fw[j]=(pC@v10_pose_Rwc(pose).T+pose[:3][None,:]).astype(np.float32)
        # Fill a possible initial no-camera prefix with first valid world pose only for
        # storage; samples without causal camera pose are still excluded later.
        good=np.isfinite(fw).all((1,2));
        if good.any(): fw[:np.flatnonzero(good)[0]]=fw[np.flatnonzero(good)[0]]
        t=np.asarray(d['time'],np.float64); dt=np.maximum(np.diff(t),1e-5); fv=np.zeros_like(fw); fv[1:]=(fw[1:]-fw[:-1])/dt[:,None,None]; fv[0]=fv[1]
        np.savez_compressed(cp,foot_base=fb,foot_world=fw,foot_vel_world=fv)
    V10_FOOT_BASE[ri]=fb; V10_FOOT_WORLD[ri]=fw; V10_FOOT_VEL_WORLD[ri]=fv

# Causal derivatives used by the temporal model and physics losses.
V10_DDQ={}; V10_DGYRO={}
for ri,rec in enumerate(run_records):
    t=np.asarray(rec['data']['time'],np.float64); dq=np.asarray(rec['data']['P'][:,:,3:6],np.float64)
    dt=np.maximum(np.diff(t),1e-5); ddq=np.zeros_like(dq); ddq[1:]=(dq[1:]-dq[:-1])/dt[:,None,None]; ddq[0]=ddq[1]
    V10_DDQ[ri]=np.clip(ddq,-500,500).astype(np.float32)
    g=np.asarray(rec['data']['B'][:,3:6],np.float64); dg=np.zeros_like(g); dg[1:]=(g[1:]-g[:-1])/dt[:,None]; dg[0]=dg[1]
    V10_DGYRO[ri]=np.clip(dg,-500,500).astype(np.float32)

print('\nV10-0 READY')
print('Raw runs:',[(r['terrain_type'],str(r['base'])) for r in run_records])
print('Visual streams attached:',[(r['terrain_type'],sorted([k for k in r.get('visual',{}) if k in ['DINO','CONV','GEOM','FLOW']])) for r in run_records])

# %% notebook cell 8
# ======================================================================
# V10-1 — DENSE CAUSAL RGB-D GEOMETRY / SURFEL MEMORY
# 2-cm voxel resolution, normals, RGB, source time/frame/grid, optional seg.
# ======================================================================
from scipy.spatial import cKDTree
from collections import Counter
import numpy as np, pandas as pd, cv2, math, os, json
from PIL import Image

V10_VOXEL_M=0.02
V10_PIXEL_STRIDE=4
V10_MAX_POINTS_PER_FRAME=5500
V10_QUERY_RADIUS_M=0.18
V10_QUERY_POINTS=32
V10_GEOM_STATS_DIM=18


def _find_seg_dir(base):
    c=[]
    for p in base.iterdir():
        n=p.name.lower()
        if p.is_dir() and ('seg' in n or 'label' in n) and ('front' in n or 'lcam' in n): c.append(p)
    return c[0] if c else None

def _load_seg_for_fid(segdir,fid):
    if segdir is None: return None
    cand=list(segdir.glob(f'{fid}*'))
    if not cand: return None
    im=cv2.imread(str(cand[0]),cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.ndim==3: im=im[...,0]
    return im.astype(np.int32)


def _sample_frame_surfels(rec,frame_idx,cam='lcam_front'):
    man=rec['manifest']; base=rec['base']; fid=str(man['frame_ids'][frame_idx]); frame_num=int(man['frame_nums'][frame_idx]) if 'frame_nums' in man else int(fid)
    dep_path=base/f'depth_{cam}'/f'{fid}_{cam}_depth.png'; rgb_path=base/f'image_{cam}'/f'{fid}_{cam}.png'; pose_path=base/f'pose_{cam}.txt'
    if not dep_path.exists() or not rgb_path.exists() or not pose_path.exists(): return None
    depth=v10_read_depth(dep_path); rgb=np.asarray(Image.open(rgb_path).convert('RGB'))
    allposes=rec.setdefault('_v10_pose_cache',{}).get(cam)
    if allposes is None:
        allposes=np.loadtxt(pose_path).astype(np.float64); rec['_v10_pose_cache'][cam]=allposes
    if frame_num>=len(allposes): return None
    pose=np.asarray(allposes[frame_num],np.float64); front_pose=np.asarray(man['poses'][frame_idx],np.float64)
    seg=_load_seg_for_fid(rec.get('_v10_segdir'),fid) if cam=='lcam_front' else None
    ys=np.arange(1,depth.shape[0]-1,V10_PIXEL_STRIDE); xs=np.arange(1,depth.shape[1]-1,V10_PIXEL_STRIDE); vv,uu=np.meshgrid(ys,xs,indexing='ij'); Z=depth[vv,uu]
    valid=np.isfinite(Z)&(Z>.25)&(Z<8.0); x=Z; y=(uu-CX)*Z/FX; z=(vv-CY)*Z/FY; pc=np.stack([x,y,z],-1).astype(np.float32)
    dx=np.zeros_like(pc); dy=np.zeros_like(pc); dx[:,1:-1]=pc[:,2:]-pc[:,:-2]; dy[1:-1,:]=pc[2:]-pc[:-2]
    nc=np.cross(dx,dy); nc/=np.maximum(np.linalg.norm(nc,axis=-1,keepdims=True),1e-6)
    Rwc=v10_pose_Rwc(pose); world_vec=pc@Rwc.T; world=world_vec+pose[:3][None,None,:]
    # Robot/front-heading local volume, irrespective of which teacher camera saw it.
    local=(world-front_pose[:3][None,None,:])@v10_heading_Rlw(front_pose).T; lx,ly,lz=local[...,0],local[...,1],local[...,2]
    valid &= (lx>-1.5)&(lx<4.5)&(np.abs(ly)<2.7)&(lz>-2.2)&(lz<1.2)
    ids=np.flatnonzero(valid.ravel())
    if len(ids)==0: return None
    wf=world.reshape(-1,3)[ids].astype(np.float32); nw=(nc.reshape(-1,3)[ids]@Rwc.T).astype(np.float32); nw/=np.maximum(np.linalg.norm(nw,axis=1,keepdims=True),1e-6); rf=rgb[vv,uu].reshape(-1,3)[ids].astype(np.uint8)
    uf=uu.ravel()[ids]; vf=vv.ravel()[ids]
    if cam=='lcam_front':
        gx=np.clip((uf.astype(np.float32)/CAM_W*RAW_GRID).astype(np.int16),0,RAW_GRID-1); gy=np.clip((vf.astype(np.float32)/CAM_H*RAW_GRID).astype(np.int16),0,RAW_GRID-1); grid=(gy*RAW_GRID+gx).astype(np.int16)
    else: grid=np.full(len(ids),-1,np.int16)
    sem=np.full(len(ids),-1,np.int16) if seg is None else np.clip(seg[vv,uu].ravel()[ids],-1,32767).astype(np.int16)
    vox=np.floor(wf/V10_VOXEL_M).astype(np.int64); key=vox[:,0]*73856093 ^ vox[:,1]*19349663 ^ vox[:,2]*83492791; _,keep=np.unique(key,return_index=True)
    if len(keep)>V10_MAX_POINTS_PER_FRAME:
        rng=np.random.default_rng(V10_SEED+int(frame_idx)+sum(map(ord,cam))); keep=rng.choice(keep,V10_MAX_POINTS_PER_FRAME,replace=False)
    return wf[keep],nw[keep],rf[keep],grid[keep],sem[keep]


def v10_build_geometry_memory(rec,force=False):
    env,run=rec['environment'],rec['run']; root=V10_GEOM_ROOT/f'{env}_{run}'
    root.mkdir(parents=True,exist_ok=True)
    fields=['xyz','normal','rgb','time','frame','grid','sem']
    if (not force) and all((root/f'{k}.npy').exists() for k in fields):
        print('Reusing dense geometry memory',env,run); return root
    rec['_v10_segdir']=_find_seg_dir(rec['base'])
    chunks={k:[] for k in fields}; man=rec['manifest']; N=len(man['frame_ids'])
    print(f'Building {env}/{run} dense 2-cm geometry memory from {N} RGB-D frames')
    cams=V10_TEACHER_CAMERAS if V10_USE_ALL_CAMERAS else ['lcam_front']
    for j in range(N):
        for cam in cams:
            out=_sample_frame_surfels(rec,j,cam)
            if out is None: continue
            xyz,n,rgb,grid,sem=out; m=len(xyz)
            chunks['xyz'].append(xyz); chunks['normal'].append(n); chunks['rgb'].append(rgb)
            chunks['time'].append(np.full(m,float(man['camera_times'][j]),np.float32)); chunks['frame'].append(np.full(m,j,np.int32)); chunks['grid'].append(grid); chunks['sem'].append(sem)
        if (j+1)%100==0: print(' ',j+1,'/',N,'points so far',sum(len(x) for x in chunks['xyz']))
    if not chunks['xyz']: raise RuntimeError('No geometry points produced for '+env+'/'+run)
    arr={k:np.concatenate(v,axis=0) for k,v in chunks.items()}
    for k,v in arr.items(): np.save(root/f'{k}.npy',v)
    meta=dict(voxel_m=V10_VOXEL_M,pixel_stride=V10_PIXEL_STRIDE,points=int(len(arr['xyz'])),seg_dir=str(rec.get('_v10_segdir')))
    (root/'meta.json').write_text(json.dumps(meta,indent=2))
    print(' saved',len(arr['xyz']),'surfels ->',root)
    return root


class V10DenseGeometryMemory:
    def __init__(self,root):
        self.root=Path(root)
        self.xyz=np.load(self.root/'xyz.npy',mmap_mode='r'); self.normal=np.load(self.root/'normal.npy',mmap_mode='r')
        self.rgb=np.load(self.root/'rgb.npy',mmap_mode='r'); self.time=np.load(self.root/'time.npy',mmap_mode='r')
        self.frame=np.load(self.root/'frame.npy',mmap_mode='r'); self.grid=np.load(self.root/'grid.npy',mmap_mode='r'); self.sem=np.load(self.root/'sem.npy',mmap_mode='r')
        self.tree=cKDTree(np.asarray(self.xyz[:,:2],np.float32))
    def _empty(self):
        return dict(points=np.zeros((V10_QUERY_POINTS,9),np.float32),stats=np.zeros(V10_GEOM_STATS_DIM,np.float32),pairs=np.zeros((0,2),np.int32),sem_hist=np.zeros(16,np.float32),ids=np.zeros(0,np.int64))
    def query(self,world_xyz,query_t,pose7,radius=V10_QUERY_RADIUS_M,max_events=256):
        world_xyz=np.asarray(world_xyz,np.float64)
        ids=np.asarray(self.tree.query_ball_point(world_xyz[:2],r=float(radius)),np.int64)
        if len(ids)==0: return self._empty()
        ids=ids[np.asarray(self.time[ids])<=float(query_t)+1e-9]
        if len(ids)==0: return self._empty()
        d=np.linalg.norm(np.asarray(self.xyz[ids],np.float64)-world_xyz[None,:],axis=1)
        order=np.argsort(d)[:max_events]; ids=ids[order]; d=d[order]
        # Prefer independent recent frames when choosing visual evidence, but preserve
        # all nearby surfels for geometric fitting.
        Rlw=v10_heading_Rlw(pose7); rel=(np.asarray(self.xyz[ids],np.float64)-world_xyz[None,:])@Rlw.T
        nl=np.asarray(self.normal[ids],np.float64)@Rlw.T; rgb=np.asarray(self.rgb[ids],np.float32)/255.0
        z=rel[:,2]; xy=rel[:,:2]
        # Quadratic local height fit z=ax^2+by^2+cxy+dx+ey+f.
        curvature=0.; ridge=0.; rough=float(np.std(z));
        if len(ids)>=8:
            X=np.stack([xy[:,0]**2,xy[:,1]**2,xy[:,0]*xy[:,1],xy[:,0],xy[:,1],np.ones(len(xy))],1)
            try:
                coef=np.linalg.lstsq(X,z,rcond=None)[0]; res=z-X@coef; rough=float(np.std(res))
                H=np.array([[2*coef[0],coef[2]],[coef[2],2*coef[1]]],np.float64); ev=np.linalg.eigvalsh(H)
                curvature=float(np.mean(np.abs(ev))); ridge=float(np.max(np.abs(ev)))
            except Exception: pass
        normal_mean=nl.mean(0); normal_mean/=max(np.linalg.norm(normal_mean),1e-6)
        slope=float(np.arccos(np.clip(abs(normal_mean[2]),0,1)))
        step=float(np.quantile(z,.95)-np.quantile(z,.05)) if len(z)>2 else 0.
        age=np.maximum(0,float(query_t)-np.asarray(self.time[ids],np.float64))
        stats=np.array([
            1.0,min(len(ids)/128.,1.0),float(d[0]),float(age.mean()),float(z.mean()),float(z.std()),float(z.max()-z.min()),
            float(normal_mean[0]),float(normal_mean[1]),float(normal_mean[2]),slope,rough,curvature,ridge,step,
            float(np.std(xy[:,0])),float(np.std(xy[:,1])),float(np.mean(np.linalg.norm(xy,axis=1)))
        ],np.float32)
        # Deterministic nearest point set.
        k=min(V10_QUERY_POINTS,len(ids)); pick=np.arange(k); pts=np.concatenate([rel[pick],nl[pick],rgb[pick]],axis=1).astype(np.float32)
        if k<V10_QUERY_POINTS: pts=np.concatenate([pts,np.zeros((V10_QUERY_POINTS-k,9),np.float32)],0)
        # frame/grid pairs used to recover local DINO/Conv/WAFT evidence.
        pairs=[]; seen=set()
        for ii in ids:
            gr=int(self.grid[ii])
            if gr<0: continue
            p=(int(self.frame[ii]),gr)
            if p not in seen: seen.add(p); pairs.append(p)
            if len(pairs)>=12: break
        sem=np.asarray(self.sem[ids],np.int64); sem=sem[sem>=0]; sh=np.zeros(16,np.float32)
        if len(sem):
            for s in sem: sh[int(s)%16]+=1
            sh/=max(sh.sum(),1)
        return dict(points=pts,stats=stats,pairs=np.asarray(pairs,np.int32),sem_hist=sh,ids=ids)
    def cloud(self,world_xyz,query_t,pose7,radius=1.0,max_points=8000):
        ids=np.asarray(self.tree.query_ball_point(np.asarray(world_xyz)[:2],r=float(radius)),np.int64)
        ids=ids[np.asarray(self.time[ids])<=float(query_t)+1e-9]
        if len(ids)>max_points:
            rng=np.random.default_rng(V10_SEED); ids=rng.choice(ids,max_points,replace=False)
        rel=(np.asarray(self.xyz[ids],np.float64)-np.asarray(world_xyz)[None,:])@v10_heading_Rlw(pose7).T
        return rel.astype(np.float32),np.asarray(self.rgb[ids])

V10_GEOMETRY={}
for ri,rec in enumerate(run_records):
    root=v10_build_geometry_memory(rec,force=False)
    V10_GEOMETRY[ri]=V10DenseGeometryMemory(root)
    print(rec['terrain_type'],'dense surfels=',len(V10_GEOMETRY[ri].xyz))

print('V10-1 READY')

# %% notebook cell 9
# ======================================================================
# V10-2 — CAUSAL PHYSICAL MEMORY + MATERIAL FEATURES + TEMPORAL DATASET
# ======================================================================
from scipy.spatial import cKDTree
import numpy as np, pandas as pd, json, math, random, os
import torch, torch.nn as nn

V10_CONTEXT_SEC=1.0
V10_HISTORY_DT=0.02       # 50 Hz history tokens
V10_SAMPLE_HOP_SEC=0.10   # query every 100 ms
V10_HORIZONS=np.round(np.arange(.05,1.001,.05),2).astype(np.float32)  # 20 horizons
V10_NH=len(V10_HORIZONS)
V10_LEG_NAMES=['FL','FR','RL','RR']
V10_SPLIT=dict(train_end=.60,val_start=.65,val_end=.80,test_start=.85)
V10_MATERIAL_DIM=176       # DINO64 + Conv64 + Flow32 + semantic-hist16
V10_PHYS_DIM=18
V10_HISTORY_TOKENS=int(round(V10_CONTEXT_SEC/V10_HISTORY_DT))

# ----------------------------------------------------------------------
# Frozen V7 branch adapters provide a stronger starting representation than raw
# random projections. We reuse ONLY the trained local DINO/Conv/WAFT adapters;
# the new raw RGB-D geometry branch is kept completely separate.
class _V7LocalAdapter(nn.Module):
    def __init__(self,din):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,256),nn.GELU(),nn.LayerNorm(256))
    def forward(self,x): return self.net(x)
class _V7FlowTemporal(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(6,64,3,padding=1),nn.GELU(),nn.Conv1d(64,128,3,padding=1),nn.GELU()); self.out=nn.Linear(128,256)
    def forward(self,x):
        B,K,S,C=x.shape; y=x.permute(0,2,3,1).reshape(B*S,C,K); y=self.net(y)[:,:,-1]; return self.out(y).reshape(B,S,256)

def _load_prefixed(module,sd,prefix):
    sub={k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)}
    if not sub: return False
    module.load_state_dict(sub,strict=True); return True

V10_V7_SHORT=Path(OUT_ROOT)/'V7_PRIOR_SHORT.pt'
V10_USE_TRAINED_VIS_ADAPTERS=V10_V7_SHORT.exists() and all('visual' in r for r in run_records)
V10_ADAPTED_MATERIAL={}
if V10_USE_TRAINED_VIS_ADAPTERS:
    try:
        sd=torch.load(V10_V7_SHORT,map_location='cpu',weights_only=True)
    except TypeError:
        sd=torch.load(V10_V7_SHORT,map_location='cpu')
    rec0=run_records[0]; dinoA=_V7LocalAdapter(rec0['visual']['DINO'].shape[-1]); convA=_V7LocalAdapter(rec0['visual']['CONV'].shape[-1]); flowT=_V7FlowTemporal(); flowA=_V7LocalAdapter(256)
    ok=(_load_prefixed(dinoA.net,sd,'terrain_encoder.dino.adapter.') and _load_prefixed(convA.net,sd,'terrain_encoder.conv.adapter.') and
        _load_prefixed(flowT,sd,'terrain_encoder.flow_temporal.') and _load_prefixed(flowA.net,sd,'terrain_encoder.flow.adapter.'))
    V10_USE_TRAINED_VIS_ADAPTERS=bool(ok)
    if ok:
        dinoA=dinoA.to(DEVICE).eval(); convA=convA.to(DEVICE).eval(); flowT=flowT.to(DEVICE).eval(); flowA=flowA.to(DEVICE).eval()
        rng=np.random.default_rng(V10_SEED); Rd=torch.as_tensor((rng.standard_normal((256,64))/8.).astype(np.float32),device=DEVICE); Rc=torch.as_tensor((rng.standard_normal((256,64))/8.).astype(np.float32),device=DEVICE); Rf=torch.as_tensor((rng.standard_normal((256,32))/np.sqrt(32)).astype(np.float32),device=DEVICE)
        with torch.inference_mode():
            for ri,rec in enumerate(run_records):
                env,run=rec['environment'],rec['run']; cp=V10_DATA_ROOT/f'{env}_{run}_V10_MATERIAL_TOKENS.npy'
                if cp.exists():
                    V10_ADAPTED_MATERIAL[ri]=np.load(cp,mmap_mode='r'); continue
                N=len(rec['visual']['DINO']); mm=np.lib.format.open_memmap(cp,mode='w+',dtype=np.float16,shape=(N,64,160))
                for st in range(0,N,32):
                    en=min(N,st+32); d=torch.as_tensor(np.asarray(rec['visual']['DINO'][st:en],np.float32),device=DEVICE); c=torch.as_tensor(np.asarray(rec['visual']['CONV'][st:en],np.float32),device=DEVICE); fl=torch.as_tensor(np.asarray(rec['visual']['FLOW'][st:en],np.float32),device=DEVICE)
                    zd=dinoA(d)@Rd; zc=convA(c)@Rc; zf=flowA(flowT(fl))@Rf; mm[st:en]=torch.cat([zd,zc,zf],-1).cpu().numpy().astype(np.float16)
                mm.flush(); V10_ADAPTED_MATERIAL[ri]=np.load(cp,mmap_mode='r')
        del dinoA,convA,flowT,flowA; torch.cuda.empty_cache()
print('Using trained V7 local visual adapters:',V10_USE_TRAINED_VIS_ADAPTERS)

class V10MaterialProjector:
    def __init__(self,rec,ri,seed=V10_SEED):
        self.ri=int(ri); self.available=('visual' in rec and all(k in rec['visual'] for k in ['DINO','CONV','FLOW']))
        self.adapted=V10_ADAPTED_MATERIAL.get(self.ri,None)
        if self.adapted is not None: return
        if not self.available: self.Rd=self.Rc=self.Rf=None; return
        d=rec['visual']['DINO'].shape[-1]; c=rec['visual']['CONV'].shape[-1]; f=rec['visual']['FLOW'].shape[1]*rec['visual']['FLOW'].shape[-1]
        rng=np.random.default_rng(seed); self.Rd=(rng.standard_normal((d,64))/np.sqrt(64)).astype(np.float32); self.Rc=(rng.standard_normal((c,64))/np.sqrt(64)).astype(np.float32); self.Rf=(rng.standard_normal((f,32))/np.sqrt(32)).astype(np.float32)
    def aggregate(self,rec,pairs,sem_hist):
        out=np.zeros(V10_MATERIAL_DIM,np.float32)
        if len(pairs) and self.adapted is not None:
            vals=[]
            for fr,gr in np.asarray(pairs,np.int64):
                if 0<=fr<len(self.adapted): vals.append(np.asarray(self.adapted[fr,gr],np.float32))
            if vals: out[:160]=np.mean(vals,0)
        elif self.available and len(pairs):
            ds=[]; cs=[]; fs=[]
            for fr,gr in np.asarray(pairs,np.int64):
                if fr<0 or fr>=len(rec['visual']['DINO']): continue
                ds.append(np.asarray(rec['visual']['DINO'][fr,gr],np.float32)@self.Rd); cs.append(np.asarray(rec['visual']['CONV'][fr,gr],np.float32)@self.Rc); fs.append(np.asarray(rec['visual']['FLOW'][fr,:,gr,:],np.float32).reshape(-1)@self.Rf)
            if ds: out[:64]=np.mean(ds,0); out[64:128]=np.mean(cs,0); out[128:160]=np.mean(fs,0)
        out[160:176]=np.asarray(sem_hist,np.float32); return out

V10_MATERIAL={ri:V10MaterialProjector(rec,ri,V10_SEED+ri) for ri,rec in enumerate(run_records)}

# ----------------------------------------------------------------------
# Causal haptic/proprio memory. Past contacts only; no model residual is used yet.
# ----------------------------------------------------------------------
class V10PhysicalMemory:
    def __init__(self,rec,ri,hop_sec=.04):
        d=rec['data']; t=np.asarray(d['time']); dt=float(np.median(np.diff(t))); stride=max(1,int(round(hop_sec/dt)))
        xyz=[]; tm=[]; F=[]; E=[]; slip=[]; leg=[]
        for j in range(2,len(t),stride):
            fw=V10_FOOT_WORLD[ri][j]
            if not np.isfinite(fw).all(): continue
            for l in range(4):
                if d['contact'][j,l]<.5: continue
                v=V10_FOOT_VEL_WORLD[ri][j,l]
                xyz.append(fw[l]); tm.append(float(t[j])); F.append(d['P'][j,l,9:12]); E.append(d['E'][j,l]); slip.append(float(np.linalg.norm(v[:2]))); leg.append(l)
        self.xyz=np.asarray(xyz,np.float32); self.time=np.asarray(tm,np.float32); self.F=np.asarray(F,np.float32); self.E=np.asarray(E,np.float32)
        self.slip=np.asarray(slip,np.float32); self.leg=np.asarray(leg,np.int8); self.tree=cKDTree(self.xyz[:,:2]) if len(self.xyz) else None
    def query(self,world_xyz,query_t,radius=.22,max_events=64):
        if self.tree is None: return np.zeros(V10_PHYS_DIM,np.float32)
        ids=np.asarray(self.tree.query_ball_point(np.asarray(world_xyz)[:2],r=float(radius)),np.int64)
        ids=ids[self.time[ids]<float(query_t)-1e-6]
        if len(ids)==0: return np.zeros(V10_PHYS_DIM,np.float32)
        d=np.linalg.norm(self.xyz[ids]-np.asarray(world_xyz)[None,:],axis=1); order=np.argsort(d)[:max_events]; ids=ids[order]; d=d[order]
        age=np.maximum(0,float(query_t)-self.time[ids]); w=np.exp(-d/.10)*np.exp(-age/4.0); w/=max(w.sum(),1e-8)
        Fm=(self.F[ids]*w[:,None]).sum(0); Fs=np.sqrt(((self.F[ids]-Fm)**2*w[:,None]).sum(0)+1e-6)
        Em=(self.E[ids]*w[:,None]).sum(0); Es=np.sqrt(((self.E[ids]-Em)**2*w[:,None]).sum(0)+1e-6)
        lm=np.bincount(self.leg[ids].astype(int),weights=w,minlength=4).astype(np.float32)
        return np.concatenate([Fm,Fs,Em,Es,[float((self.slip[ids]*w).sum()),min(len(ids)/32.,1.),float((age*w).sum()),float((d*w).sum())],lm[:2]]).astype(np.float32)[:V10_PHYS_DIM]

V10_PHYSICAL={ri:V10PhysicalMemory(rec,ri) for ri,rec in enumerate(run_records)}

# ----------------------------------------------------------------------
# Helpers for history features and strict temporal split.
# ----------------------------------------------------------------------
def _run_split(rec,tq):
    t=rec['data']['time']; t0=float(t[0]); T=float(t[-1]-t[0]); f=(float(tq)-t0)/max(T,1e-6)
    if f<=V10_SPLIT['train_end']: return 'train'
    if V10_SPLIT['val_start']<=f<=V10_SPLIT['val_end']: return 'val'
    if f>=V10_SPLIT['test_start']: return 'test'
    return None

def _leg_history(rec,ri,idx,hist_ids):
    d=rec['data']; P=d['P'][hist_ids].transpose(1,0,2); L=d['L_LEG'][hist_ids].transpose(1,0,2); E=d['E'][hist_ids].transpose(1,0,2)
    U=np.broadcast_to(d['U'][hist_ids][None],(4,len(hist_ids),6)); C=d['contact'][hist_ids].T[:,:,None]
    return np.concatenate([P,L,E,U,C],axis=-1).astype(np.float32)  # 25

def _body_history(rec,hist_ids):
    d=rec['data']; return np.concatenate([d['B'][hist_ids],d['U'][hist_ids]],axis=-1).astype(np.float32) #18

# ----------------------------------------------------------------------
# Build/cache the full temporal dataset. Targets include force for stage 2, but the
# proposal model never receives future force.
# ----------------------------------------------------------------------
V10_TEMPORAL_CACHE=V10_DATA_ROOT/'V10_TEMPORAL_DATASET.npz'
V10_TEMPORAL_META=V10_DATA_ROOT/'V10_TEMPORAL_META.csv'
if V10_TEMPORAL_CACHE.exists() and V10_TEMPORAL_META.exists():
    z=np.load(V10_TEMPORAL_CACHE,allow_pickle=False)
    V10_DATA={k:z[k] for k in z.files}; V10_META=pd.read_csv(V10_TEMPORAL_META)
    print('Reusing temporal dataset',len(V10_META))
else:
    arrays={k:[] for k in ['leg_hist','body_hist','future_ctrl','y_foot','y_contact','y_state9','y_body','y_force','y_E','y_slip','y_ddq','y_dgyro','y_foot_base']}; meta=[]
    for ri,rec in enumerate(run_records):
        d=rec['data']; t=np.asarray(d['time']); dt=float(np.median(np.diff(t))); hs=max(1,int(round(V10_HISTORY_DT/dt))); Hsteps=int(round(V10_CONTEXT_SEC/dt)); qhop=max(1,int(round(V10_SAMPLE_HOP_SEC/dt)))
        offs=np.asarray([max(1,int(round(float(h)/dt))) for h in V10_HORIZONS],np.int64)
        for idx in range(Hsteps,len(t)-int(offs.max())-1,qhop):
            split=_run_split(rec,t[idx]);
            if split is None: continue
            hist_ids=idx-hs*np.arange(V10_HISTORY_TOKENS-1,-1,-1,dtype=np.int64)
            if hist_ids[0]<0: continue
            pose,_=v10_pose_at(rec,idx)
            if pose is None: continue
            lh=_leg_history(rec,ri,idx,hist_ids); bh=_body_history(rec,hist_ids)
            fc=[]; yf=[]; yc=[]; ys=[]; yb=[]; yF=[]; yE=[]; ysl=[]; yddq=[]; ydg=[]; yfb=[]
            good=True
            for h,off in zip(V10_HORIZONS,offs):
                j=idx+int(off); fw=V10_FOOT_WORLD[ri][j]
                if not np.isfinite(fw).all(): good=False; break
                loc=v10_heading_from_world(fw,pose).astype(np.float32)
                us=d['U'][idx:j+1]; ls=d['L_LEG'][idx:j+1]
                fc.append(np.concatenate([us[-1],ls[-1].reshape(-1),us.mean(0),ls.mean(0).reshape(-1)]).astype(np.float32))
                yf.append(loc); yc.append(d['contact'][j]); ys.append(d['P'][j,:,:9]); yb.append(d['B'][j]); yF.append(d['P'][j,:,9:12]); yE.append(d['E'][j])
                yfb.append(V10_FOOT_BASE[ri][j])
                ysl.append(np.linalg.norm(V10_FOOT_VEL_WORLD[ri][j,:,:2],axis=-1).astype(np.float32)); yddq.append(V10_DDQ[ri][j]); ydg.append(V10_DGYRO[ri][j])
            if not good: continue
            for k,v in [('leg_hist',lh),('body_hist',bh),('future_ctrl',fc),('y_foot',yf),('y_contact',yc),('y_state9',ys),('y_body',yb),('y_force',yF),('y_E',yE),('y_slip',ysl),('y_ddq',yddq),('y_dgyro',ydg),('y_foot_base',yfb)]: arrays[k].append(np.asarray(v,np.float32))
            meta.append(dict(ri=ri,terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],idx=idx,time=float(t[idx]),split=split))
        print(rec['terrain_type'],'samples so far',len(meta))
    V10_DATA={k:np.asarray(v,np.float32) for k,v in arrays.items()}; V10_META=pd.DataFrame(meta)
    np.savez_compressed(V10_TEMPORAL_CACHE,**V10_DATA); V10_META.to_csv(V10_TEMPORAL_META,index=False)

V10_TRAIN=np.flatnonzero(V10_META['split'].to_numpy()=='train'); V10_VAL=np.flatnonzero(V10_META['split'].to_numpy()=='val'); V10_TEST=np.flatnonzero(V10_META['split'].to_numpy()=='test')
print('V10 temporal shapes:');
for k,v in V10_DATA.items(): print(' ',k,v.shape)
print('split sizes',len(V10_TRAIN),len(V10_VAL),len(V10_TEST))
print('V10-2 READY')

# %% notebook cell 10
# ======================================================================
# V10-3 — TEMPORAL 4-LEG COUPLED MOTION / TOUCHDOWN PROPOSAL TRANSFORMER
# ======================================================================
import copy, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import precision_recall_fscore_support

V10_PROP_ROOT=V10_MODEL_ROOT/'proposal'; V10_PROP_ROOT.mkdir(parents=True,exist_ok=True)
V10_PROP_BATCH=48; V10_PROP_EPOCHS=120; V10_PROP_PATIENCE=18; V10_PROP_LR=3e-4

# Train-only normalization.
def _mean_std(x,ids,axes):
    y=x[ids]; return y.mean(axis=axes).astype(np.float32),np.maximum(y.std(axis=axes),1e-4).astype(np.float32)
LEG_MEAN,LEG_STD=_mean_std(V10_DATA['leg_hist'],V10_TRAIN,(0,1,2)); BODY_MEAN,BODY_STD=_mean_std(V10_DATA['body_hist'],V10_TRAIN,(0,1)); CTRL_MEAN,CTRL_STD=_mean_std(V10_DATA['future_ctrl'],V10_TRAIN,(0,1))
FOOT_MEAN,FOOT_STD=_mean_std(V10_DATA['y_foot'],V10_TRAIN,(0,1,2)); STATE_MEAN,STATE_STD=_mean_std(V10_DATA['y_state9'],V10_TRAIN,(0,1,2)); YB_MEAN,YB_STD=_mean_std(V10_DATA['y_body'],V10_TRAIN,(0,1))

class _PropDS(Dataset):
    def __init__(self,ids): self.ids=np.asarray(ids,np.int64)
    def __len__(self): return len(self.ids)
    def __getitem__(self,k):
        i=int(self.ids[k])
        return (torch.from_numpy(((V10_DATA['leg_hist'][i]-LEG_MEAN)/LEG_STD).astype(np.float32)),
                torch.from_numpy(((V10_DATA['body_hist'][i]-BODY_MEAN)/BODY_STD).astype(np.float32)),
                torch.from_numpy(((V10_DATA['future_ctrl'][i]-CTRL_MEAN)/CTRL_STD).astype(np.float32)),
                torch.from_numpy(((V10_DATA['y_foot'][i]-FOOT_MEAN)/FOOT_STD).astype(np.float32)),
                torch.from_numpy(V10_DATA['y_contact'][i].astype(np.float32)),
                torch.from_numpy(((V10_DATA['y_state9'][i]-STATE_MEAN)/STATE_STD).astype(np.float32)),
                torch.from_numpy(((V10_DATA['y_body'][i]-YB_MEAN)/YB_STD).astype(np.float32)))

class V10CausalEncoder(nn.Module):
    def __init__(self,din,d=128,layers=3,heads=4,maxT=64):
        super().__init__(); self.inp=nn.Linear(din,d); self.pos=nn.Parameter(torch.randn(1,maxT,d)*.01)
        layer=nn.TransformerEncoderLayer(d,heads,4*d,.10,batch_first=True,norm_first=True,activation='gelu')
        self.tx=nn.TransformerEncoder(layer,layers); self.norm=nn.LayerNorm(d)
    def forward(self,x):
        T=x.shape[1]; z=self.inp(x)+self.pos[:,:T]; mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1)
        return self.norm(self.tx(z,mask=mask)[:,-1])

class V10CoupledProposal(nn.Module):
    def __init__(self,leg_dim,body_dim,ctrl_dim):
        super().__init__(); d=128
        self.leg_enc=V10CausalEncoder(leg_dim,d,3,4); self.body_enc=V10CausalEncoder(body_dim,d,3,4); self.leg_id=nn.Embedding(4,d)
        layer=nn.TransformerEncoderLayer(d,4,4*d,.10,batch_first=True,norm_first=True,activation='gelu'); self.couple=nn.TransformerEncoder(layer,3); self.cnorm=nn.LayerNorm(d)
        self.ctrl=nn.Sequential(nn.Linear(ctrl_dim,128),nn.GELU(),nn.LayerNorm(128)); self.hz=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,128),nn.LayerNorm(128))
        self.qf=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,192),nn.GELU())
        self.contact=nn.Linear(192,1); self.foot=nn.Linear(192,3); self.state=nn.Linear(192,9)
        self.bodyq=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,192),nn.GELU(),nn.Linear(192,12))
        self.register_buffer('horizons',torch.as_tensor(V10_HORIZONS,dtype=torch.float32))
    def encode_history(self,leg_hist,body_hist):
        B,L,T,C=leg_hist.shape; z=self.leg_enc(leg_hist.reshape(B*L,T,C)).reshape(B,L,-1)
        ids=torch.arange(4,device=leg_hist.device)[None].expand(B,-1); z=z+self.leg_id(ids); b=self.body_enc(body_hist)
        tok=torch.cat([b[:,None],z],1); tok=self.cnorm(self.couple(tok)); return tok[:,1:],tok[:,0]
    def query_features(self,leg_ctx,body_ctx,future_ctrl):
        B,H,_=future_ctrl.shape; hz=self.hz(self.horizons[None,:,None].expand(B,-1,-1)); cc=self.ctrl(future_ctrl)
        lc=leg_ctx[:,None].expand(-1,H,-1,-1); bc=body_ctx[:,None,None].expand(-1,H,4,-1); hc=hz[:,:,None].expand(-1,-1,4,-1); uc=cc[:,:,None].expand(-1,-1,4,-1)
        return self.qf(torch.cat([lc,bc,hc,uc],-1)),hz,cc
    def forward(self,leg_hist,body_hist,future_ctrl):
        leg,body=self.encode_history(leg_hist,body_hist); q,hz,cc=self.query_features(leg,body,future_ctrl)
        body_out=self.bodyq(torch.cat([body[:,None].expand(-1,cc.shape[1],-1),cc,hz],-1))
        return dict(contact_logits=self.contact(q).squeeze(-1),foot=self.foot(q),state9=self.state(q),body=body_out,leg_ctx=leg,body_ctx=body)

proposal=V10CoupledProposal(V10_DATA['leg_hist'].shape[-1],V10_DATA['body_hist'].shape[-1],V10_DATA['future_ctrl'].shape[-1]).to(DEVICE)
opt=torch.optim.AdamW(proposal.parameters(),lr=V10_PROP_LR,weight_decay=2e-4)
train_dl=DataLoader(_PropDS(V10_TRAIN),batch_size=V10_PROP_BATCH,shuffle=True,num_workers=0); val_dl=DataLoader(_PropDS(V10_VAL),batch_size=64,shuffle=False,num_workers=0)

def _prop_loss(o,yfoot,yc,ystate,ybody):
    lf=F.smooth_l1_loss(o['foot'],yfoot,beta=.25); ls=F.smooth_l1_loss(o['state9'],ystate,beta=.35); lb=F.smooth_l1_loss(o['body'],ybody,beta=.35)
    pos=(yc.sum()+1e-6); pos_weight=torch.clamp((yc.numel()-yc.sum())/pos,1.,4.)
    lc=F.binary_cross_entropy_with_logits(o['contact_logits'],yc,pos_weight=pos_weight)
    return 2.5*lf+1.0*ls+.5*lb+.6*lc,dict(foot=lf,state=ls,body=lb,contact=lc)

ck=V10_PROP_ROOT/'V10_PROPOSAL.pt'
if ck.exists():
    sd=torch.load(ck,map_location=DEVICE); proposal.load_state_dict(sd['state_dict']); print('Reused proposal checkpoint',ck)
else:
    best=None; bestv=1e99; patience=0; hist=[]
    for ep in range(1,V10_PROP_EPOCHS+1):
        proposal.train(); tl=[]
        for leg,bod,ctrl,yf,yc,ys,yb in train_dl:
            leg,bod,ctrl,yf,yc,ys,yb=[x.to(DEVICE) for x in [leg,bod,ctrl,yf,yc,ys,yb]]; opt.zero_grad(set_to_none=True)
            o=proposal(leg,bod,ctrl); loss,_=_prop_loss(o,yf,yc,ys,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(proposal.parameters(),2.); opt.step(); tl.append(float(loss.detach().cpu()))
        proposal.eval(); vl=[]
        with torch.inference_mode():
            for leg,bod,ctrl,yf,yc,ys,yb in val_dl:
                leg,bod,ctrl,yf,yc,ys,yb=[x.to(DEVICE) for x in [leg,bod,ctrl,yf,yc,ys,yb]]; loss,_=_prop_loss(proposal(leg,bod,ctrl),yf,yc,ys,yb); vl.append(float(loss.cpu()))
        vm=float(np.mean(vl)); hist.append(dict(epoch=ep,train=float(np.mean(tl)),val=vm))
        if vm<bestv-1e-4: bestv=vm; best=copy.deepcopy(proposal.state_dict()); patience=0
        else: patience+=1
        if ep==1 or ep%10==0 or patience==0: print(f'proposal ep={ep:03d} train={np.mean(tl):.4f} val={vm:.4f} best={bestv:.4f}')
        if patience>=V10_PROP_PATIENCE: break
    proposal.load_state_dict(best); pd.DataFrame(hist).to_csv(V10_PROP_ROOT/'history.csv',index=False)
    torch.save(dict(state_dict=proposal.state_dict(),LEG_MEAN=LEG_MEAN,LEG_STD=LEG_STD,BODY_MEAN=BODY_MEAN,BODY_STD=BODY_STD,CTRL_MEAN=CTRL_MEAN,CTRL_STD=CTRL_STD,
                    FOOT_MEAN=FOOT_MEAN,FOOT_STD=FOOT_STD,STATE_MEAN=STATE_MEAN,STATE_STD=STATE_STD,YB_MEAN=YB_MEAN,YB_STD=YB_STD),ck)
proposal.eval()

@torch.inference_mode()
def v10_proposal_predict(ids,batch=128):
    ids=np.asarray(ids,np.int64); PF=[]; PC=[]; PS=[]; PB=[]
    for st in range(0,len(ids),batch):
        ii=ids[st:st+batch]
        leg=torch.as_tensor((V10_DATA['leg_hist'][ii]-LEG_MEAN)/LEG_STD,dtype=torch.float32,device=DEVICE)
        bod=torch.as_tensor((V10_DATA['body_hist'][ii]-BODY_MEAN)/BODY_STD,dtype=torch.float32,device=DEVICE)
        ctrl=torch.as_tensor((V10_DATA['future_ctrl'][ii]-CTRL_MEAN)/CTRL_STD,dtype=torch.float32,device=DEVICE)
        o=proposal(leg,bod,ctrl)
        PF.append(o['foot'].cpu().numpy()*FOOT_STD+FOOT_MEAN); PC.append(torch.sigmoid(o['contact_logits']).cpu().numpy()); PS.append(o['state9'].cpu().numpy()*STATE_STD+STATE_MEAN); PB.append(o['body'].cpu().numpy()*YB_STD+YB_MEAN)
    return np.concatenate(PF),np.concatenate(PC),np.concatenate(PS),np.concatenate(PB)

V10_PROP_FOOT,V10_PROP_CONTACT,V10_PROP_STATE,V10_PROP_BODY=v10_proposal_predict(np.arange(len(V10_META)))
# Proposal evaluation.
rows=[]
for hi,h in enumerate(V10_HORIZONS):
    ids=V10_TEST; err=np.linalg.norm(V10_PROP_FOOT[ids,hi]-V10_DATA['y_foot'][ids,hi],axis=-1); true=V10_DATA['y_contact'][ids,hi].reshape(-1); pred=(V10_PROP_CONTACT[ids,hi].reshape(-1)>=.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(true.astype(int),pred,average='binary',zero_division=0)
    rows.append(dict(horizon_s=float(h),foot_MAE_m=float(err.mean()),foot_p95_m=float(np.quantile(err,.95)),within_5cm=float((err<=.05).mean()),contact_precision=pr,contact_recall=re,contact_F1=f1))
V10_PROP_EVAL=pd.DataFrame(rows); display(V10_PROP_EVAL); V10_PROP_EVAL.to_csv(V10_EVAL_ROOT/'proposal_metrics.csv',index=False)
print('V10-3 READY')

# %% notebook cell 11
# ======================================================================
# V10-4 — QUERY DENSE GEOMETRY/MATERIAL/PHYSICAL MAP AT PREDICTED FEET
#          + FIT TRAIN-ONLY COUPLED PHYSICS PRIORS
# ======================================================================
from sklearn.linear_model import Ridge
import numpy as np, pandas as pd, torch, os, math, json

V10_EXT_ROOT=V10_DATA_ROOT/'predicted_foot_extero'; V10_EXT_ROOT.mkdir(parents=True,exist_ok=True)
EXT_FIELDS={
    'geo_points':(len(V10_META),V10_NH,4,V10_QUERY_POINTS,9),
    'geo_stats':(len(V10_META),V10_NH,4,V10_GEOM_STATS_DIM),
    'material':(len(V10_META),V10_NH,4,V10_MATERIAL_DIM),
    'physical':(len(V10_META),V10_NH,4,V10_PHYS_DIM),
}

def _ext_exists(): return all((V10_EXT_ROOT/f'{k}.npy').exists() for k in EXT_FIELDS)
if not _ext_exists():
    mm={k:np.lib.format.open_memmap(V10_EXT_ROOT/f'{k}.npy',mode='w+',dtype=np.float16,shape=shape) for k,shape in EXT_FIELDS.items()}
    print('Building predicted-location causal exteroception for',len(V10_META),'queries')
    for i,row in V10_META.iterrows():
        ri=int(row.ri); rec=run_records[ri]; pose,_=v10_pose_at(rec,int(row.idx)); tq=float(row.time)
        for hi in range(V10_NH):
            world=v10_world_from_heading(V10_PROP_FOOT[i,hi],pose)
            for leg in range(4):
                q=V10_GEOMETRY[ri].query(world[leg],tq,pose)
                mm['geo_points'][i,hi,leg]=q['points'].astype(np.float16); mm['geo_stats'][i,hi,leg]=q['stats'].astype(np.float16)
                mm['material'][i,hi,leg]=V10_MATERIAL[ri].aggregate(rec,q['pairs'],q['sem_hist']).astype(np.float16)
                mm['physical'][i,hi,leg]=V10_PHYSICAL[ri].query(world[leg],tq).astype(np.float16)
        if (i+1)%250==0: print(' ',i+1,'/',len(V10_META))
    for v in mm.values(): v.flush()
V10_EXT={k:np.load(V10_EXT_ROOT/f'{k}.npy',mmap_mode='r') for k in EXT_FIELDS}
print('Exteroception ready:',{k:v.shape for k,v in V10_EXT.items()})

# ----------------------------------------------------------------------
# Train-only low-order physical calibration.
# tau = [ddq,dq,sin(q),cos(q),F,1] W per leg.
# body acc = [sumF,1] Wlin.
# angular acc = [sum(r x F), omega,1] Wrot.
# ----------------------------------------------------------------------
def _ridge(X,Y,a=1e-3):
    m=Ridge(alpha=a,fit_intercept=False); m.fit(X,Y); return m.coef_.T.astype(np.float32)

invW=[]; Xlin=[]; Ylin=[]; Xrot=[]; Yrot=[]
for leg in range(4):
    X=[]; Y=[]
    for i in V10_TRAIN:
        ri=int(V10_META.iloc[i].ri); rec=run_records[ri]; idx=int(V10_META.iloc[i].idx); d=rec['data']; t0=float(V10_META.iloc[i].time)
        for h in V10_HORIZONS:
            j=idx+int(round(float(h)/np.median(np.diff(d['time'])))); j=min(j,len(d['time'])-1)
            q=d['P'][j,leg,:3]; dq=d['P'][j,leg,3:6]; ddq=V10_DDQ[ri][j,leg]; Fv=d['P'][j,leg,9:12]
            X.append(np.concatenate([ddq,dq,np.sin(q),np.cos(q),Fv,[1.]],dtype=np.float32)); Y.append(d['P'][j,leg,6:9])
    invW.append(_ridge(np.asarray(X),np.asarray(Y)))
invW=np.stack(invW)

# Fit body mappings on train target horizons (subsample for speed).
for i in V10_TRAIN[::max(1,len(V10_TRAIN)//1500)]:
    for hi in range(V10_NH):
        Fv=V10_DATA['y_force'][i,hi]; acc=V10_DATA['y_body'][i,hi,:3]; omega=V10_DATA['y_body'][i,hi,9:12] if V10_DATA['y_body'].shape[-1]>=12 else np.zeros(3)
        r=V10_DATA['y_foot_base'][i,hi]; moment=np.cross(r,Fv).sum(0)
        Xlin.append(np.concatenate([Fv.sum(0),[1.]])); Ylin.append(acc)
        Xrot.append(np.concatenate([moment,omega,[1.]])); Yrot.append(V10_DATA['y_dgyro'][i,hi])
Wlin=_ridge(np.asarray(Xlin,np.float32),np.asarray(Ylin,np.float32)); Wrot=_ridge(np.asarray(Xrot,np.float32),np.asarray(Yrot,np.float32))
V10_PHYSICS=dict(invW=invW,Wlin=Wlin,Wrot=Wrot)
np.savez(V10_MODEL_ROOT/'V10_PHYSICS_PRIOR.npz',**V10_PHYSICS)

# Physics force prior from proposal-predicted q,dq,tau. It is deliberately only a
# prior; full-body losses and the residual model may reject it.
F_PRIOR=np.zeros((len(V10_META),V10_NH,4,3),np.float32)
for i in range(len(V10_META)):
    current_dq=V10_DATA['leg_hist'][i,:, -1,3:6]
    prev_dq=current_dq.copy()
    for hi,h in enumerate(V10_HORIZONS):
        st=V10_PROP_STATE[i,hi]  # 4x9: q,dq,tau
        q=st[:,0:3]; dq=st[:,3:6]; tau=st[:,6:9]
        dh=float(V10_HORIZONS[hi]-(V10_HORIZONS[hi-1] if hi else 0.0)); ddq=(dq-prev_dq)/max(dh,1e-3); prev_dq=dq
        for l in range(4):
            W=invW[l]; other=np.concatenate([ddq[l],dq[l],np.sin(q[l]),np.cos(q[l]),[1.]],dtype=np.float32)
            # Feature layout: ddq3,dq3,sin3,cos3,F3,1.
            base=np.concatenate([ddq[l],dq[l],np.sin(q[l]),np.cos(q[l])])@W[:12] + W[15]
            WF=W[12:15]  # F(row) @ WF -> tau(row)
            F_PRIOR[i,hi,l]=(tau[l]-base)@np.linalg.pinv(WF)
F_PRIOR=np.clip(F_PRIOR,-1200,1200).astype(np.float32)
np.save(V10_DATA_ROOT/'V10_FORCE_PHYSICS_PRIOR.npy',F_PRIOR)
print('Physics prior built',F_PRIOR.shape,'mean magnitude',float(np.linalg.norm(F_PRIOR,axis=-1).mean()))
print('V10-4 READY')

# %% notebook cell 12
# ======================================================================
# V10-5 — GEOMETRY/MATERIAL-CONDITIONED COUPLED FORCE MODEL
# Physics residual + translational/rotational/inverse-dynamics constraints.
# ======================================================================
import copy, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

V10_FORCE_ROOT=V10_MODEL_ROOT/'force'; V10_FORCE_ROOT.mkdir(parents=True,exist_ok=True)
V10_FORCE_BATCH=32; V10_FORCE_EPOCHS=160; V10_FORCE_PATIENCE=24; V10_FORCE_LR=4e-4
YF=V10_DATA['y_force']; FSTD=np.maximum(YF[V10_TRAIN].std(axis=(0,1,2)),5.0).astype(np.float32)

class _ForceDS(Dataset):
    def __init__(self,ids): self.ids=np.asarray(ids,np.int64)
    def __len__(self): return len(self.ids)
    def __getitem__(self,k):
        i=int(self.ids[k])
        return dict(i=torch.tensor(i),
            leg=torch.from_numpy(((V10_DATA['leg_hist'][i]-LEG_MEAN)/LEG_STD).astype(np.float32)),
            body=torch.from_numpy(((V10_DATA['body_hist'][i]-BODY_MEAN)/BODY_STD).astype(np.float32)),
            ctrl=torch.from_numpy(((V10_DATA['future_ctrl'][i]-CTRL_MEAN)/CTRL_STD).astype(np.float32)),
            propfoot=torch.from_numpy(V10_PROP_FOOT[i].astype(np.float32)), propcontact=torch.from_numpy(V10_PROP_CONTACT[i].astype(np.float32)), propstate=torch.from_numpy(V10_PROP_STATE[i].astype(np.float32)),
            gp=torch.from_numpy(np.asarray(V10_EXT['geo_points'][i],np.float32)), gs=torch.from_numpy(np.asarray(V10_EXT['geo_stats'][i],np.float32)), mat=torch.from_numpy(np.asarray(V10_EXT['material'][i],np.float32)), phys=torch.from_numpy(np.asarray(V10_EXT['physical'][i],np.float32)),
            prior=torch.from_numpy(F_PRIOR[i]), yf=torch.from_numpy(YF[i]), yc=torch.from_numpy(V10_DATA['y_contact'][i]), yE=torch.from_numpy(V10_DATA['y_E'][i]), yslip=torch.from_numpy(V10_DATA['y_slip'][i]),
            ystate=torch.from_numpy(V10_DATA['y_state9'][i]), ybody=torch.from_numpy(V10_DATA['y_body'][i]), yddq=torch.from_numpy(V10_DATA['y_ddq'][i]), ydgyro=torch.from_numpy(V10_DATA['y_dgyro'][i]), yfootbase=torch.from_numpy(V10_DATA['y_foot_base'][i]))

class V10GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.pt=nn.Sequential(nn.Linear(9,64),nn.GELU(),nn.Linear(64,96),nn.GELU()); self.st=nn.Sequential(nn.LayerNorm(V10_GEOM_STATS_DIM),nn.Linear(V10_GEOM_STATS_DIM,64),nn.GELU()); self.out=nn.Sequential(nn.Linear(160,128),nn.GELU(),nn.LayerNorm(128))
    def forward(self,p,s):
        z=self.pt(p).amax(-2); return self.out(torch.cat([z,self.st(s)],-1))

class V10ForceModel(nn.Module):
    def __init__(self,proposal_backbone):
        super().__init__(); self.backbone=proposal_backbone
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.geo=V10GeometryEncoder(); self.mat=nn.Sequential(nn.LayerNorm(V10_MATERIAL_DIM),nn.Linear(V10_MATERIAL_DIM,192),nn.GELU(),nn.Linear(192,128),nn.GELU(),nn.LayerNorm(128))
        self.phys=nn.Sequential(nn.LayerNorm(V10_PHYS_DIM),nn.Linear(V10_PHYS_DIM,96),nn.GELU(),nn.Linear(96,64),nn.GELU())
        self.prop=nn.Sequential(nn.LayerNorm(16),nn.Linear(16,96),nn.GELU()) # foot3+contact1+state9+prior3
        self.fuse=nn.Sequential(nn.LayerNorm(192+128+128+64+96),nn.Linear(608,384),nn.GELU(),nn.Dropout(.10),nn.Linear(384,192),nn.GELU(),nn.LayerNorm(192))
        self.delta=nn.Linear(192,3); self.logsigma=nn.Linear(192,3); self.auxE=nn.Linear(192,3); self.auxSlip=nn.Linear(192,1)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias); nn.init.zeros_(self.logsigma.weight); nn.init.constant_(self.logsigma.bias,math.log(30.0))
        self.register_buffer('fstd',torch.as_tensor(FSTD))
    def forward(self,leg_hist,body_hist,ctrl,propfoot,propcontact,propstate,gp,gs,mat,phys,prior):
        with torch.no_grad(): leg_ctx,body_ctx=self.backbone.encode_history(leg_hist,body_hist); qfeat,_,_=self.backbone.query_features(leg_ctx,body_ctx,ctrl)
        g=self.geo(gp,gs); m=self.mat(mat); ph=self.phys(phys); pr=self.prop(torch.cat([propfoot,propcontact.unsqueeze(-1),propstate,prior],-1))
        z=self.fuse(torch.cat([qfeat,g,m,ph,pr],-1)); pred=prior+self.delta(z)*self.fstd
        return dict(force=pred,logsigma=self.logsigma(z).clamp(-1.0,5.5),E=self.auxE(z),slip=self.auxSlip(z).squeeze(-1),latent=z)

force_model=V10ForceModel(proposal).to(DEVICE)
opt=torch.optim.AdamW([p for p in force_model.parameters() if p.requires_grad],lr=V10_FORCE_LR,weight_decay=2e-4)
train_dl=DataLoader(_ForceDS(V10_TRAIN),batch_size=V10_FORCE_BATCH,shuffle=True,num_workers=0); val_dl=DataLoader(_ForceDS(V10_VAL),batch_size=48,shuffle=False,num_workers=0)
Wlin_t=torch.as_tensor(V10_PHYSICS['Wlin'],device=DEVICE); Wrot_t=torch.as_tensor(V10_PHYSICS['Wrot'],device=DEVICE); invW_t=torch.as_tensor(V10_PHYSICS['invW'],device=DEVICE)

def _force_loss(o,b):
    pred=o['force']; gt=b['yf']; contact=b['yc']; ae=torch.abs(pred-gt); vec=torch.linalg.vector_norm(pred-gt,dim=-1)
    w=(.35+1.65*contact).unsqueeze(-1); data=(F.smooth_l1_loss(pred,gt,beta=5.,reduction='none')*w).mean(); vector=(vec*(.35+1.65*contact)).mean()
    flat=ae.reshape(-1); k=max(1,int(.20*flat.numel())); tail=torch.topk(flat,k).values.mean()
    # Swing force should be near zero.
    zero=((1-contact)*torch.linalg.vector_norm(pred,dim=-1)).mean()
    # Translational full-body calibration.
    sf=pred.sum(2); Xa=torch.cat([sf,torch.ones((*sf.shape[:-1],1),device=DEVICE)],-1); acc=Xa@Wlin_t; llin=F.smooth_l1_loss(acc,b['ybody'][...,:3],beta=1.)
    # Rotational whole-body calibration using target lever arms and current target omega.
    mom=torch.cross(b['yfootbase'],pred,dim=-1).sum(2); omega=b['ybody'][...,9:12]; Xr=torch.cat([mom,omega,torch.ones((*mom.shape[:-1],1),device=DEVICE)],-1); alpha=Xr@Wrot_t; lrot=F.smooth_l1_loss(alpha,b['ydgyro'],beta=2.)
    # Per-leg inverse dynamics. Feature layout ddq,dq,sin(q),cos(q),F,1.
    q=b['ystate'][...,:3]; dq=b['ystate'][...,3:6]; tau=b['ystate'][...,6:9]; ddq=b['yddq']; ones=torch.ones((*q.shape[:-1],1),device=DEVICE)
    X=torch.cat([ddq,dq,torch.sin(q),torch.cos(q),pred,ones],-1); tauhat=torch.einsum('bhlf,lfj->bhlj',X,invW_t); linv=F.smooth_l1_loss(tauhat,tau,beta=2.)
    lE=F.smooth_l1_loss(o['E'],b['yE'],beta=2.); lsl=F.smooth_l1_loss(o['slip'],b['yslip'],beta=.05)
    sigma=torch.exp(o['logsigma']); nll=(.5*((pred-gt)/sigma)**2+o['logsigma']).mean()
    total=data+.12*vector+.22*tail+.03*zero+.05*llin+.04*lrot+.06*linv+.03*lE+.02*lsl+.02*nll
    return total,dict(data=data,vector=vector,tail=tail,lin=llin,rot=lrot,inv=linv,E=lE,slip=lsl,nll=nll)

ck=V10_FORCE_ROOT/'V10_FORCE_MODEL.pt'
if ck.exists():
    sd=torch.load(ck,map_location=DEVICE); force_model.load_state_dict(sd['state_dict']); print('Reused force model checkpoint',ck)
else:
    best=None; bestscore=1e99; patience=0; hist=[]
    for ep in range(1,V10_FORCE_EPOCHS+1):
        force_model.train(); tl=[]
        for b in train_dl:
            for k in b:
                if torch.is_tensor(b[k]): b[k]=b[k].to(DEVICE)
            opt.zero_grad(set_to_none=True); o=force_model(b['leg'],b['body'],b['ctrl'],b['propfoot'],b['propcontact'],b['propstate'],b['gp'],b['gs'],b['mat'],b['phys'],b['prior']); loss,_=_force_loss(o,b); loss.backward(); torch.nn.utils.clip_grad_norm_(force_model.parameters(),2.); opt.step(); tl.append(float(loss.detach().cpu()))
        force_model.eval(); errs=[]; vl=[]
        with torch.inference_mode():
            for b in val_dl:
                for k in b:
                    if torch.is_tensor(b[k]): b[k]=b[k].to(DEVICE)
                o=force_model(b['leg'],b['body'],b['ctrl'],b['propfoot'],b['propcontact'],b['propstate'],b['gp'],b['gs'],b['mat'],b['phys'],b['prior']); loss,_=_force_loss(o,b); vl.append(float(loss.cpu())); errs.append(torch.abs(o['force']-b['yf']).cpu().numpy().reshape(-1))
        e=np.concatenate(errs); score=float(e.mean()+.40*np.quantile(e,.95)); hist.append(dict(epoch=ep,train=float(np.mean(tl)),val_loss=float(np.mean(vl)),val_component_mean=float(e.mean()),val_component_p95=float(np.quantile(e,.95)),score=score))
        if score<bestscore-1e-3: bestscore=score; best=copy.deepcopy(force_model.state_dict()); patience=0
        else: patience+=1
        if ep==1 or ep%10==0 or patience==0: print(f'force ep={ep:03d} train={np.mean(tl):.3f} val_comp={e.mean():.3f} p95={np.quantile(e,.95):.3f} score={score:.3f}')
        if patience>=V10_FORCE_PATIENCE: break
    force_model.load_state_dict(best); pd.DataFrame(hist).to_csv(V10_FORCE_ROOT/'history.csv',index=False)
    torch.save(dict(state_dict=force_model.state_dict(),FSTD=FSTD),ck)
force_model.eval()

@torch.inference_mode()
def v10_force_predict(ids,batch=64):
    dl=DataLoader(_ForceDS(ids),batch_size=batch,shuffle=False,num_workers=0); out=[]
    for b in dl:
        for k in b:
            if torch.is_tensor(b[k]): b[k]=b[k].to(DEVICE)
        o=force_model(b['leg'],b['body'],b['ctrl'],b['propfoot'],b['propcontact'],b['propstate'],b['gp'],b['gs'],b['mat'],b['phys'],b['prior']); out.append(o['force'].cpu().numpy())
    return np.concatenate(out)

V10_FORCE_PRED=v10_force_predict(np.arange(len(V10_META)))
np.save(V10_EVAL_ROOT/'V10_FORCE_PRED_ALL.npy',V10_FORCE_PRED)
print('V10-5 READY',V10_FORCE_PRED.shape)

# %% notebook cell 13
# ======================================================================
# V10-6 — STRICT HELD-OUT 0..1 s EVALUATION
# Per component, per foot, per horizon; foot placement + risk diagnostics.
# ======================================================================
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

ids=V10_TEST; gt=V10_DATA['y_force'][ids]; pr=V10_FORCE_PRED[ids]; ae=np.abs(pr-gt); ve=np.linalg.norm(pr-gt,axis=-1)
rows=[]
for hi,h in enumerate(V10_HORIZONS):
    for leg in range(4):
        for ci,cname in enumerate(['Fx','Fy','Fz']):
            e=ae[:,hi,leg,ci]
            rows.append(dict(horizon_s=float(h),leg=V10_LEG_NAMES[leg],component=cname,n=len(e),MAE_N=float(e.mean()),median_N=float(np.median(e)),p90_N=float(np.quantile(e,.90)),p95_N=float(np.quantile(e,.95)),p99_N=float(np.quantile(e,.99)),max_N=float(e.max()),within_5N=float((e<=5.).mean()),p95_pass_5N=bool(np.quantile(e,.95)<=5.),literal_max_pass_5N=bool(e.max()<=5.)))
V10_COMPONENT_METRICS=pd.DataFrame(rows)

vr=[]
for hi,h in enumerate(V10_HORIZONS):
    for leg in range(4):
        e=ve[:,hi,leg]
        vr.append(dict(horizon_s=float(h),leg=V10_LEG_NAMES[leg],mean_vector_N=float(e.mean()),p95_vector_N=float(np.quantile(e,.95)),max_vector_N=float(e.max()),within_5_vector=float((e<=5.).mean())))
V10_VECTOR_METRICS=pd.DataFrame(vr)

# Foot / contact metrics from proposal.
fr=[]
for hi,h in enumerate(V10_HORIZONS):
    err=np.linalg.norm(V10_PROP_FOOT[ids,hi]-V10_DATA['y_foot'][ids,hi],axis=-1)
    for leg in range(4):
        yt=V10_DATA['y_contact'][ids,hi,leg].astype(int); yp=(V10_PROP_CONTACT[ids,hi,leg]>=.5).astype(int); p,r,f1,_=precision_recall_fscore_support(yt,yp,average='binary',zero_division=0)
        fr.append(dict(horizon_s=float(h),leg=V10_LEG_NAMES[leg],foot_mean_cm=float(100*err[:,leg].mean()),foot_p95_cm=float(100*np.quantile(err[:,leg],.95)),within_2cm=float((err[:,leg]<=.02).mean()),within_5cm=float((err[:,leg]<=.05).mean()),contact_precision=p,contact_recall=r,contact_F1=f1))
V10_FOOT_METRICS=pd.DataFrame(fr)

V10_COMPONENT_METRICS.to_csv(V10_EVAL_ROOT/'force_component_metrics.csv',index=False); V10_VECTOR_METRICS.to_csv(V10_EVAL_ROOT/'force_vector_metrics.csv',index=False); V10_FOOT_METRICS.to_csv(V10_EVAL_ROOT/'foot_metrics.csv',index=False)

print('='*120); print('V10 FORCE COMPONENT METRICS — HELD-OUT TEST'); display(V10_COMPONENT_METRICS)
print('='*120); print('V10 FOOT PLACEMENT METRICS'); display(V10_FOOT_METRICS)

# Compact hard-gate table.
gate=(V10_COMPONENT_METRICS.groupby(['leg','component']).agg(worst_horizon_p95_N=('p95_N','max'),worst_horizon_MAE_N=('MAE_N','max'),min_within5=('within_5N','min')).reset_index())
gate['p95_all_horizons_pass']=gate['worst_horizon_p95_N']<=5.; display(gate); gate.to_csv(V10_EVAL_ROOT/'HARD_5N_GATE.csv',index=False)
ALL_COMPONENT_P95_PASS=bool((V10_COMPONENT_METRICS['p95_N']<=5.).all()); ALL_COMPONENT_LITERAL_MAX_PASS=bool((V10_COMPONENT_METRICS['max_N']<=5.).all())
print('\nHARD RESULT:')
print('Every leg/component/horizon p95 <= 5 N:',ALL_COMPONENT_P95_PASS)
print('Literal maximum error <= 5 N everywhere:',ALL_COMPONENT_LITERAL_MAX_PASS)

# Horizon curves, one distinct plot per component.
for cname in ['Fx','Fy','Fz']:
    fig,ax=plt.subplots(figsize=(9,4))
    for leg in V10_LEG_NAMES:
        q=V10_COMPONENT_METRICS[(V10_COMPONENT_METRICS.leg==leg)&(V10_COMPONENT_METRICS.component==cname)]
        ax.plot(q.horizon_s,q.p95_N,marker='o',label=leg)
    ax.axhline(5.,linestyle='--',label='5 N target'); ax.set_xlabel('horizon [s]'); ax.set_ylabel(f'{cname} p95 abs error [N]'); ax.set_title(f'V10 held-out {cname} p95 error'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(V10_EVAL_ROOT/f'{cname}_p95_by_horizon.png',dpi=170); plt.show()
print('V10-6 COMPLETE ->',V10_EVAL_ROOT)

# %% notebook cell 14
# ======================================================================
# V10-7 — ONLINE PROPRIOCEPTIVE CORRECTION + SHADOW FOOTHOLD PLANNER
# Chronological residual-map replay, local vertical twin, per-leg XY risk maps.
# ======================================================================
from collections import defaultdict
import numpy as np,pandas as pd,torch,matplotlib.pyplot as plt,math

# ----------------------------------------------------------------------
# Online residual memory: score first, then update. Fixed 0.10-s horizon avoids
# the same future contact appearing many times and leaking into itself.
# ----------------------------------------------------------------------
class V10OnlineResidualMap:
    def __init__(self,res=.05,alpha=.20): self.res=float(res); self.alpha=float(alpha); self.cells={}
    def key(self,p): p=np.asarray(p); return tuple(np.floor(p/self.res).astype(int).tolist())
    def query(self,p):
        k=self.key(p); vals=[]; ws=[]
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                kk=(k[0]+dx,k[1]+dy,k[2]);
                if kk in self.cells:
                    r,n=self.cells[kk]; w=1/(1+dx*dx+dy*dy); vals.append(r*w); ws.append(w)
        return np.sum(vals,0)/sum(ws) if vals else np.zeros(3,np.float32)
    def update(self,p,residual,confidence=1.0):
        k=self.key(p); residual=np.asarray(residual,np.float32); a=self.alpha*float(np.clip(confidence,.1,1.))
        if k not in self.cells: self.cells[k]=(residual.copy(),1)
        else:
            old,n=self.cells[k]; self.cells[k]=((1-a)*old+a*residual,n+1)

hidx=int(np.argmin(np.abs(V10_HORIZONS-.10)))
replay=[]
for ri,rec in enumerate(run_records):
    ids=V10_TEST[V10_META.iloc[V10_TEST].ri.to_numpy()==ri]; ids=ids[np.argsort(V10_META.iloc[ids].time.to_numpy())]; mem=V10OnlineResidualMap()
    for i in ids:
        row=V10_META.iloc[i]; pose,_=v10_pose_at(rec,int(row.idx)); world=v10_world_from_heading(V10_PROP_FOOT[i,hidx],pose)
        for leg in range(4):
            raw=V10_FORCE_PRED[i,hidx,leg]; gt=V10_DATA['y_force'][i,hidx,leg]; corr=mem.query(world[leg]); adapted=raw+corr
            replay.append(dict(terrain=rec['terrain_type'],time=float(row.time),leg=V10_LEG_NAMES[leg],raw_Fx=raw[0],raw_Fy=raw[1],raw_Fz=raw[2],adapt_Fx=adapted[0],adapt_Fy=adapted[1],adapt_Fz=adapted[2],gt_Fx=gt[0],gt_Fy=gt[1],gt_Fz=gt[2],raw_vec=float(np.linalg.norm(raw-gt)),adapt_vec=float(np.linalg.norm(adapted-gt))))
        # Update only after scoring this chronological query, using target contacts.
        for leg in range(4):
            if V10_DATA['y_contact'][i,hidx,leg]>.5:
                raw=V10_FORCE_PRED[i,hidx,leg]; gt=V10_DATA['y_force'][i,hidx,leg]; En=float(np.linalg.norm(V10_DATA['y_E'][i,hidx,leg])); conf=1/(1+.05*En)
                mem.update(world[leg],gt-raw,conf)
V10_ONLINE=pd.DataFrame(replay); V10_ONLINE.to_csv(V10_EVAL_ROOT/'online_residual_replay_0p1s.csv',index=False)
print('Online replay raw mean vector error:',V10_ONLINE.raw_vec.mean(),'N')
print('Online replay adapted mean vector error:',V10_ONLINE.adapt_vec.mean(),'N')

# ----------------------------------------------------------------------
# Single-query force head for shadow candidate evaluation.
# Candidate geometry is real/casual; candidate force response is not a paired
# simulator counterfactual, so planner results are explicitly SHADOW only.
# ----------------------------------------------------------------------
@torch.inference_mode()
def _planner_predict_candidate(sample_i,hi,leg,candidate_local,candidate_world,geo_q):
    rec=run_records[int(V10_META.iloc[sample_i].ri)]; ri=int(V10_META.iloc[sample_i].ri); tq=float(V10_META.iloc[sample_i].time)
    mat=V10_MATERIAL[ri].aggregate(rec,geo_q['pairs'],geo_q['sem_hist']); phys=V10_PHYSICAL[ri].query(candidate_world,tq)
    # Build one normal sample, then replace only the queried leg/horizon extero/foot.
    b=_ForceDS([sample_i])[0]
    gp=b['gp'][None].to(DEVICE); gs=b['gs'][None].to(DEVICE); ma=b['mat'][None].to(DEVICE); ph=b['phys'][None].to(DEVICE); pf=b['propfoot'][None].to(DEVICE)
    gp[:,hi,leg]=torch.as_tensor(geo_q['points'],device=DEVICE); gs[:,hi,leg]=torch.as_tensor(geo_q['stats'],device=DEVICE); ma[:,hi,leg]=torch.as_tensor(mat,device=DEVICE); ph[:,hi,leg]=torch.as_tensor(phys,device=DEVICE); pf[:,hi,leg]=torch.as_tensor(candidate_local,device=DEVICE)
    o=force_model(b['leg'][None].to(DEVICE),b['body'][None].to(DEVICE),b['ctrl'][None].to(DEVICE),pf,b['propcontact'][None].to(DEVICE),b['propstate'][None].to(DEVICE),gp,gs,ma,ph,b['prior'][None].to(DEVICE))
    return o['force'][0,hi,leg].cpu().numpy(),np.exp(o['logsigma'][0,hi,leg].cpu().numpy())


def v10_shadow_plan(sample_i,leg,horizon=.40,span=.12,step=.04):
    hi=int(np.argmin(np.abs(V10_HORIZONS-float(horizon)))); row=V10_META.iloc[int(sample_i)]; ri=int(row.ri); rec=run_records[ri]; pose,_=v10_pose_at(rec,int(row.idx)); tq=float(row.time)
    nominal=V10_PROP_FOOT[sample_i,hi,leg].copy(); offsets=np.arange(-span,span+1e-9,step); rows=[]
    for dx in offsets:
        for dy in offsets:
            cand=nominal.copy(); cand[0]+=dx; cand[1]+=dy; world=v10_world_from_heading(cand[None],pose)[0]; q=V10_GEOMETRY[ri].query(world,tq,pose)
            # Snap candidate height to the causal reconstructed surface when observed.
            if q['stats'][0]>.5:
                cand[2]+=float(q['stats'][4]); world=v10_world_from_heading(cand[None],pose)[0]; q=V10_GEOMETRY[ri].query(world,tq,pose)
            Fp,sig=_planner_predict_candidate(sample_i,hi,leg,cand,world,q); s=q['stats']; available=s[0]
            slope=min(s[10]/.7,2.); rough=min(s[11]/.08,2.); ridge=min(s[13]/8.,2.); stepv=min(s[14]/.18,2.)
            force_mag=np.linalg.norm(Fp); tang=np.linalg.norm(Fp[:2])/max(abs(Fp[2]),10.); unc=float(np.mean(sig)); reach=np.linalg.norm([dx,dy])/max(span,1e-6)
            risk=(.22*min(force_mag/350.,2.)+.15*min(tang,2.)+.16*slope+.14*rough+.10*ridge+.08*stepv+.08*min(unc/20.,2.)+.07*reach + (1-available)*1.5)
            rows.append(dict(dx=dx,dy=dy,x_local=cand[0],y_local=cand[1],z_local=cand[2],world_x=world[0],world_y=world[1],world_z=world[2],Fx=Fp[0],Fy=Fp[1],Fz=Fp[2],sigma=float(unc),slope=float(s[10]),rough=float(s[11]),ridge=float(s[13]),step=float(s[14]),available=float(available),risk=float(risk)))
    df=pd.DataFrame(rows); best=df.iloc[df.risk.argmin()].copy(); return df,best

# Four random strict-test query examples, one terrain each.
rng=np.random.default_rng(V10_SEED); PLAN_CASES=[]
for terrain in V10_META.iloc[V10_TEST].terrain_type.unique():
    cand=V10_TEST[V10_META.iloc[V10_TEST].terrain_type.to_numpy()==terrain]; PLAN_CASES.append(int(rng.choice(cand)))
plan_summ=[]
for si in PLAN_CASES:
    row=V10_META.iloc[si]; ri=int(row.ri); rec=run_records[ri]; pose,_=v10_pose_at(rec,int(row.idx)); fw=v10_foot_world(rec,int(row.idx)); center=np.asarray(pose[:3]) if fw is None else fw.mean(0)
    # Vertical local digital twin.
    cloud,rgb=V10_GEOMETRY[ri].cloud(center,float(row.time),pose,radius=1.2)
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection='3d'); n=min(len(cloud),6000); ids=np.arange(n)
    ax.scatter(cloud[ids,0],cloud[ids,1],cloud[ids,2],s=2,alpha=.35); ax.set_xlabel('forward [m]'); ax.set_ylabel('right [m]'); ax.set_zlabel('up [m]'); ax.set_title(f"{row.terrain_type} dense causal local twin @ {row.time:.2f}s"); fig.tight_layout(); fig.savefig(V10_PLAN_ROOT/f'{row.terrain_type}_local_twin.png',dpi=170); plt.show()
    for leg in range(4):
        df,best=v10_shadow_plan(si,leg,.40); df.to_csv(V10_PLAN_ROOT/f'{row.terrain_type}_{V10_LEG_NAMES[leg]}_risk.csv',index=False)
        piv=df.pivot(index='dy',columns='dx',values='risk').sort_index(ascending=False); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(piv.values,aspect='auto'); ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{x:.2f}' for x in piv.columns],rotation=45); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f'{y:.2f}' for y in piv.index]); ax.set_xlabel('forward offset dx [m]'); ax.set_ylabel('right offset dy [m]'); ax.set_title(f"{row.terrain_type} {V10_LEG_NAMES[leg]} shadow foothold risk"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(V10_PLAN_ROOT/f'{row.terrain_type}_{V10_LEG_NAMES[leg]}_risk.png',dpi=170); plt.show()
        plan_summ.append(dict(terrain=row.terrain_type,time=float(row.time),leg=V10_LEG_NAMES[leg],best_x_local=float(best.x_local),best_y_local=float(best.y_local),best_z_local=float(best.z_local),best_world_x=float(best.world_x),best_world_y=float(best.world_y),best_world_z=float(best.world_z),pred_Fx=float(best.Fx),pred_Fy=float(best.Fy),pred_Fz=float(best.Fz),risk=float(best.risk)))
V10_PLAN_SUMMARY=pd.DataFrame(plan_summ); display(V10_PLAN_SUMMARY); V10_PLAN_SUMMARY.to_csv(V10_PLAN_ROOT/'SHADOW_FOOTHOLD_RECOMMENDATIONS.csv',index=False)
print('\nV10-7 COMPLETE')
print('IMPORTANT: planner is a shadow counterfactual until paired simulator/hardware alternative-footstep labels exist.')
