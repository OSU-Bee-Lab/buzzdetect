import json
import os

# Lives in the output folder. Records the settings that determine the schema and
# resumability of the result files written there, so a later run can't silently
# append incompatible data (e.g. a different neuron set) to existing partials.
FNAME_MANIFEST = 'buzzdetect_manifest.json'

# fields that must match for a run to safely write into an existing output folder
KEYS_LOCKED = ('modelname', 'output_mode', 'classes_out', 'precision', 'framehop_prop')


def build_manifest(modelname, framehop_prop, precision, classes_out):
    output_mode = 'detections' if precision is not None else 'activations'
    return {
        'modelname': modelname,
        'output_mode': output_mode,
        # only meaningful for activations; sorted so order of selection doesn't matter
        'classes_out': sorted(classes_out) if output_mode == 'activations' else None,
        'precision': precision,
        'framehop_prop': framehop_prop,
    }


def read_manifest(dir_out):
    path = os.path.join(dir_out, FNAME_MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def write_manifest(dir_out, manifest):
    os.makedirs(dir_out, exist_ok=True)
    path = os.path.join(dir_out, FNAME_MANIFEST)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def diff_manifests(existing, current):
    """Return a list of human-readable conflicts between two manifests."""
    conflicts = []
    for key in KEYS_LOCKED:
        old = existing.get(key)
        new = current.get(key)
        # classes_out is order-insensitive
        if key == 'classes_out' and old is not None and new is not None:
            if set(old) != set(new):
                added = sorted(set(new) - set(old))
                removed = sorted(set(old) - set(new))
                parts = []
                if added:
                    parts.append(f"added {', '.join(added)}")
                if removed:
                    parts.append(f"removed {', '.join(removed)}")
                conflicts.append(f"output classes differ ({'; '.join(parts)})")
        elif old != new:
            conflicts.append(f"{key}: existing={old!r}, requested={new!r}")
    return conflicts


def check_or_write_manifest(dir_out, manifest):
    """
    Reconcile a run's settings with any manifest already in the output folder.

    Returns (ok, message). If no manifest exists, writes one and returns ok=True.
    If one exists and matches, returns ok=True. If it conflicts, returns ok=False
    with a message explaining the mismatch; nothing is written.
    """
    existing = read_manifest(dir_out)
    if existing is None:
        write_manifest(dir_out, manifest)
        return True, None

    conflicts = diff_manifests(existing, manifest)
    if conflicts:
        msg = (
            f"Results have already been written to '{dir_out}' using different settings, so new "
            f"results would be incompatible with the existing files:\n  - "
            + "\n  - ".join(conflicts)
            + "\nEither match the existing settings, or choose an empty output folder."
        )
        return False, msg

    return True, None
