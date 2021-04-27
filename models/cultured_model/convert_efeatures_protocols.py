# Get the output of bluepyefe and convert it in the json format needed by the evaluator and model
import json

protocols_of_interest = ["firepattern_120", "IV_-100", "APWaveform_260"]

in_protocol_path = "../../efeatures_extraction/efeatures/protocols.json"
in_efeatures_path = "../../efeatures_extraction/efeatures/features.json"

out_protocol_path = "./protocols.json"
out_efeatures_path = "./features.json"

in_protocols = json.load(open(in_protocol_path, 'r'))
in_efeatures = json.load(open(in_efeatures_path, 'r'))

out_protocols = {}
out_efeatures = {"soma": {}}

for protocol_name in protocols_of_interest:

    if protocol_name in in_protocols and protocol_name in in_efeatures:

        # Convert the format of the protocols
        stimuli = [
            in_protocols[protocol_name]['holding'],
            in_protocols[protocol_name]['step']
        ]
        out_protocols[protocol_name] = {'stimuli': stimuli}

        # Convert the format of the efeatures
        efeatures_def = {}
        for feature in in_efeatures[protocol_name]['soma']:
            efeatures_def[feature['feature']] = feature['val']
        out_efeatures['soma'][protocol_name] = {'soma': efeatures_def}

s = json.dumps(out_protocols, indent=2)
with open(out_protocol_path, "w") as fp:
    fp.write(s)

s = json.dumps(out_efeatures, indent=2)
with open(out_efeatures_path, "w") as fp:
    fp.write(s)
