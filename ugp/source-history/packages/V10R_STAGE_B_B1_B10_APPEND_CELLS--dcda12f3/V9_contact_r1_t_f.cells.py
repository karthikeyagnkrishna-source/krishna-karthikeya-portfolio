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
