import json
import matplotlib.pyplot as plt

import bluepyopt as bpopt
import bluepyopt.ephys as ephys

import model
import evaluator
import time
import neuron
import plotting
import MEAutility as mu
from pprint import pprint
import numpy as np

import sys
import shutil

from pathlib import Path
import os

sys.path.append('../ecode/')

from ecode import generate_ecode_protocols, compute_rheobase_for_model, run_ecode_protocols, \
        save_intracellular_responses, save_extracellular_template

probe_type = "planar" #'linear' #
model_name = "hay_ais" # "hay_ais" #"hallermann" #'hay'
model_folder = (Path(".") / f"{model_name}_model").absolute()

# output folder for data
output_folder = Path(f"../data/{model_name}_ecode_probe_{probe_type}")

cell = model.create(model_name=model_name, release=True)

probe = model.define_electrode(probe_type=probe_type)
# probe = probe_type #None

param_names = [param.name for param in cell.params.values() if not param.frozen]
if model_name == "hallermann":
    cvode_active = False
else:
    cvode_active = True
sim = ephys.simulators.LFPySimulator(cell, cvode_active=cvode_active, electrode=probe)

# rheobase, rheo_protocols, rheo_responses = compute_rheobase_for_model(cell, sim=sim, step_min=5.34, step_max=6, step_increment=0.01)
rheobase, rheo_protocols, rheo_responses = compute_rheobase_for_model(cell, sim=sim, step_min=0.2, 
                                                                      step_increment=0.005)

ecode_protocols = generate_ecode_protocols(rheobase_current=rheobase, record_extra=True,
                                           protocols_with_lfp="firepattern")

responses_dict = run_ecode_protocols(protocols=ecode_protocols, cell=cell, sim=sim, 
                                     resample_rate_khz=40)

save_intracellular_responses(responses_dict=responses_dict, output_folder=output_folder)

eap, locations = save_extracellular_template(responses=responses_dict["firepattern"], 
                                             protocols=ecode_protocols, protocol_name="firepattern",
                                             probe=probe, output_folder=output_folder, sweep_id=1, 
                                             resample_rate_khz=20, fcut=[300, 6000],
                                             filt_type="filtfilt")

sys.path.append('../efeatures_extraction')

efeatures_output_directory = Path(f"../data/{model_name}_ecode_probe_{probe_type}/efeatures")

from bluepyefe.extract import read_recordings, extract_efeatures_at_targets, compute_rheobase,\
    group_efeatures, create_feature_protocol_files
from bluepyefe.plotting import plot_all_recordings_efeatures

from extraction_tools import build_model_metadata, model_csv_reader, get_targets, ecodes_model_timings

files_metadata = build_model_metadata(cell_id=model_name, ephys_dir=output_folder)
pprint(files_metadata[model_name])

cells = read_recordings(
    files_metadata=files_metadata,
    recording_reader=model_csv_reader
)

# define target features for different protocols
# targets = get_targets(ecodes_model_timings)
from bluepyefe.extract import convert_legacy_targets
targets = get_targets(ecodes_model_timings)
targets = convert_legacy_targets(targets)
    
pprint(targets)

# t_start = time.time()
# extract_efeatures_at_targets(
#     cells=cells, 
#     targets=targets,
# )
# t_stop = time.time()
# print(f"Elapsed time {t_stop - t_start}")

t_start = time.time()
cells = extract_efeatures_at_targets(
    cells, 
    targets,
)
t_stop = time.time()
print(f"Elapsed time {t_stop - t_start}")

compute_rheobase(
    cells, 
    protocols_rheobase=['IDthres']
)

protocols = group_efeatures(cells, targets, use_global_rheobase=True)

efeatures, protocol_definitions, current = create_feature_protocol_files(
    cells,
    protocols,
    output_directory=efeatures_output_directory,
    threshold_nvalue_save=1,
    write_files=True,
)

from extraction_tools import convert_to_bpo_format, append_extrafeatures_to_json, compute_extra_features

protocols_of_interest = ["firepattern_200", "IV_-100", "APWaveform_260"]

in_protocol_path = efeatures_output_directory / "protocols.json"
in_efeatures_path = efeatures_output_directory / "features.json"

out_protocol_path = efeatures_output_directory / "protocols_BPO_test.json"
out_efeatures_path = efeatures_output_directory / "features_BPO_test.json"

protocols_dict, efeatures_dict = convert_to_bpo_format(in_protocol_path, in_efeatures_path, 
                                                       out_protocol_path, out_efeatures_path, 
                                                       protocols_of_interest=protocols_of_interest, 
                                                       std_from_mean=0.2)

# append MEA.LFP features
eap = np.load(output_folder / "extracellular" / "template.npy")
fs = np.load(output_folder / "extracellular" / "fs.npy")

extra_features = compute_extra_features(eap, fs, upsample=10)

pprint(extra_features)

efeatures_dict = append_extrafeatures_to_json(extra_features, protocol_name="firepattern_200",
                                              efeatures_path=out_efeatures_path)

