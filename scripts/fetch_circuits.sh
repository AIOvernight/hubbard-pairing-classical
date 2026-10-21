#!/usr/bin/env bash
# Download the authors' Guppy circuit files attached to the Quantum Advantage
# Tracker submission "Superconducting pairing correlations" and verify their
# SHA256 checksums.  The files are not redistributed with this repository.
#
# Usage: scripts/fetch_circuits.sh [target_dir]      (default: data/circuits)
set -euo pipefail
dest="${1:-data/circuits}"
mkdir -p "$dest"
base="https://github.com/user-attachments/files"
while read -r id name sha; do
  out="$dest/$name"
  if [ ! -s "$out" ]; then
    echo "downloading $name"
    curl -fsSL --retry 3 -o "$out" "$base/$id/$name"
  fi
  echo "$sha  $out" | sha256sum -c -
done <<LIST
33709192 superconducting_pairing_2steps.py 0fd2e92e03d15970eb9a7e9e0f643128a9c5dff42811a45c186beb9dd45de0ac
33709184 superconducting_pairing_3steps.py 77ea24ab86545774d038bc5ec47b613b39b1ff0bb8e230e2ff68a1aa2144a108
33708662 superconducting_pairing_4steps.py 4eb1d05038b633aff64746666891dc16d5fbcaa0be030790209e0f462c52eddf
LIST
