import os
import neuron
import LFPy


def load_hay_model(model_folder, dt, tstart, tstop):
    cwd = os.getcwd()
    os.chdir(model_folder)
    cell = None
    try:
        ##define cell parameters used as input to cell-class
        cellParameters = {
            'morphology': 'morphologies/cell1.asc',
            'templatefile': ['models/L5PCbiophys3.hoc',
                             'models/L5PCtemplate.hoc'],
            'templatename': 'L5PCtemplate',
            'templateargs': 'morphologies/cell1.asc',
            'passive': False,
            'nsegs_method': None,
            'dt': dt,
            'tstart': tstart,
            'tstop': tstop,
            'v_init': -70,
            'celsius': 34,
            'pt3d': True,
        }

        # delete old sections and load compiled mechs from the mod-folder
        LFPy.cell.neuron.h("forall delete_section()")
        # Initialize cell instance, using the LFPy.Cell class
        neuron.load_mechanisms('mod')
        cell = LFPy.TemplateCell(**cellParameters)
    except:
        print('Failed to load Hay model. Wrong folder?')

    os.chdir(cwd)
    return cell

