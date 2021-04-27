from targets import targets, ecode_to_index, ecodes_timings
from pathlib import Path


def build_wcp_metadata(cell_id, ephys_dir, repetition_as_different_cells=True,
                       liquid_junction_potential=14.):

    files_metadata  = {}

    if not repetition_as_different_cells:
        files_metadata[cell_id] = {}

    for repetition in range(2, 5):

        if repetition_as_different_cells:
            current_id = f"{cell_id}_rep{repetition}"
            files_metadata[current_id] = {}
        else:
            current_id = cell_id

        for ecode in ecode_to_index:

            file_path = Path(ephys_dir) / f"{cell_id}_run{repetition}.{ecode_to_index[ecode]}.wcp"

            if not file_path.is_file():
                print(f"Missing trace {file_path}")
                continue

            metadata= {
                "filepath": str(file_path),
                "i_unit": "pA",
                "t_unit": "s",
                "v_unit": "mV",
                "ljp": liquid_junction_potential
            }

            metadata.update(ecodes_timings[ecode])

            if ecode not in files_metadata[current_id]:
                files_metadata[current_id][ecode] = [metadata]
            else:
                files_metadata[current_id][ecode].append(metadata)

    return files_metadata


def build_model_metadata(cell_id, ephys_dir, repetition_as_different_cells):
    pass