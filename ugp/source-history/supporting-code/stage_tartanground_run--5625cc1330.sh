#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 ENVIRONMENT RUN" >&2
    exit 2
fi

environment=$1
run=$2
storage_root=${V7_STORAGE_ROOT:-/path/to/ugp/V7F_4080_Run/storage}
archive_dir="$storage_root/relay_archives/$environment/$run"
parent="$storage_root/raw_tartanground/$environment/Data_anymal"
incoming="$parent/.${run}.incoming"
target="$parent/$run"
archives=(imu.zip rosbags.zip image_lcam_front.zip depth_lcam_front.zip metadata.zip)

is_complete() {
    local candidate=$1
    [[ -d "$candidate/imu" ]] &&
    [[ -f "$candidate/rosbags/$run.bag" ]] &&
    [[ -d "$candidate/image_lcam_front" ]] &&
    [[ -d "$candidate/depth_lcam_front" ]] &&
    [[ -f "$candidate/pose_lcam_front.txt" ]]
}

if is_complete "$target"; then
    echo "ALREADY_READY $environment/$run"
    exit 0
fi
if [[ -e "$target" ]]; then
    echo "Refusing to overwrite incomplete target: $target" >&2
    exit 3
fi

# Validate every archive before exposing or changing a trajectory directory.
for archive in "${archives[@]}"; do
    path="$archive_dir/$archive"
    if [[ ! -f "$path" ]]; then
        echo "Missing archive: $path" >&2
        exit 4
    fi
    unzip -tq "$path" >/dev/null
done

mkdir -p "$incoming"
for archive in "${archives[@]}"; do
    unzip -q -o "$archive_dir/$archive" -d "$incoming"
done

if ! is_complete "$incoming"; then
    echo "Extracted trajectory is incomplete: $incoming" >&2
    exit 5
fi

mv -- "$incoming" "$target"
echo "STAGED_READY $environment/$run"
