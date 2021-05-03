from targets import targets
from pathlib import Path


def build_wcp_metadata(cell_id, ephys_dir, repetition_as_different_cells=True,
                       liquid_junction_potential=14.):

    files_metadata = {}

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

            metadata.update(ecodes_wcp_timings[ecode])

            if ecode not in files_metadata[current_id]:
                files_metadata[current_id][ecode] = [metadata]
            else:
                files_metadata[current_id][ecode].append(metadata)

    return files_metadata


def build_model_metadata(cell_id, ephys_dir):
    files_metadata = {}

    files_metadata[cell_id] = {}

    ephys_dir = Path(ephys_dir)

    for protocol_folder in ephys_dir.iterdir():
        if "extracellular" not in protocol_folder.name:

            metadata = {
                "folderpath": str(protocol_folder),
                "i_unit": "nA",
                "t_unit": "ms",
                "v_unit": "mV",
            }
            ecode = protocol_folder.name
            metadata.update(ecodes_model_timings[ecode])
            files_metadata[cell_id][ecode] = [metadata]

    return files_metadata


#### Ecode params ###

ecode_to_index = {
    "IDthres": 0,
    "firepattern": 1,
    "IV": 2,
    "IDrest": 3,
    "APWaveform": 4,
    "HyperDepol": 5,
    "sAHP": 6,
    "PosCheops": 7
}

ecodes_wcp_timings = {
    "IDthres": {
        'ton': 200,
        'toff': 470
    },
    "firepattern": {
        'ton': 500,
        'toff': 4100
    },
    "IV": {
        'ton': 250,
        'toff': 3250
    },
    "IDrest": {
        'ton': 200,
        'toff': 1550
    },
    "APWaveform": {
        'ton': 150,
        'toff': 200
    },
    "HyperDepol": {
        'ton': 200,
        'toff': 920,
        'tmid': 650
    },
    "sAHP": {
        'ton': 200,
        'toff': 1125,
        'tmid': 450,
        'tmid2': 675
    },
    "PosCheops": {
        'ton': 1000,
        't1': 9000,
        't2': 10500,
        't3': 14500,
        't4': 16000,
        'toff': 18660
    }
}

ecodes_model_timings = {
    "IDthres": {
        'ton': 250,
        'toff': 520
    },
    "firepattern": {
        'ton': 250,
        'toff': 3850
    },
    "IV": {
        'ton': 250,
        'toff': 3250
    },
    "IDrest": {
        'ton': 250,
        'toff': 1600
    },
    "APWaveform": {
        'ton': 250,
        'toff': 300
    },
    "HyperDepol": {
        'ton': 250,
        'toff': 970,
        'tmid': 700
    },
    "sAHP": {
        'ton': 250,
        'toff': 1175,
        'tmid': 500,
        'tmid2': 725
    },
    "PosCheops": {
        'ton': 250,
        't1': 8250,
        't2': 9750,
        't3': 13750,
        't4': 15250,
        'toff': 17910
    }
}
