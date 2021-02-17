import logging

logger = logging.getLogger(__name__)


def replace_axon_with_hillock(sim=None, icell=None):
    """Replace axon"""

    """
    Replace axon by an hillock, an AIS and a long myelinated axon.
    """
    L_hillock = 10
    L_AIS = 35  # define the desired length of AIS
    L_total = L_hillock + L_AIS

    seg_length_to_get_diameters = 1  # the length of the segments to get the diameters
    final_seg_length = 5  # the final length of the segments during the simulation
    print('yooooo')
    diams = []
    lens = []
    seg_distance = []

    dist_seg_from_soma = 0
    for section in icell.axonal:
        L = section.L
        nseg = 1 + int(L / seg_length_to_get_diameters)  # nseg to get diameter
        section.nseg = nseg
        real_seg_length = L / nseg

        for seg in section:
            diams.append(seg.diam)
            lens.append(L / nseg)
            seg_distance.append(dist_seg_from_soma)
            dist_seg_from_soma = dist_seg_from_soma + real_seg_length
            if dist_seg_from_soma >= L_total:
                break
        if dist_seg_from_soma >= L_total:
            break

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create hillock", icell)
    sim.neuron.h.execute("create ais", icell)

    icell.hillockal.append(sec=icell.hillock)
    icell.all.append(sec=icell.hillock)

    icell.axon_initial_segment.append(sec=icell.ais)
    icell.all.append(sec=icell.ais)

    n_seg_hillock = 0
    n_seg_AIS = 0
    for dist_segment in seg_distance:
        if dist_segment <= L_hillock:
            n_seg_hillock = n_seg_hillock + 1
        else:
            n_seg_AIS = n_seg_AIS + 1

    count = 0

    icell.hillock.L = L_hillock
    icell.hillock.nseg = n_seg_hillock
    for seg in icell.hillock:
        seg.diam = diams[count]
        count = count + 1

    icell.ais.L = L_AIS
    icell.ais.nseg = n_seg_AIS
    for seg in icell.ais:
        seg.diam = diams[count]
        count = count + 1

    icell.hillock.nseg = 1 + int(L_hillock / final_seg_length)
    icell.ais.nseg = 1 + int(L_AIS / final_seg_length)

    # childsec.connect(parentsec, parentx, childx)
    icell.hillock.connect(icell.soma[0], 1.0, 0.0)
    icell.ais.connect(icell.hillock, 1.0, 0.0)

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = diams[count - 1]
    icell.myelin[0].connect(icell.ais, 1.0, 0.0)

    diams_hillock = []
    diams_AIS = []
    for seg in icell.hillock:
        diams_hillock.append(seg.diam)
    for seg in icell.ais:
        diams_AIS.append(seg.diam)

    logger.debug(
        "Replace axon with hillock of length %f and AIS of length %f, diameters are %s for the hillock and %s for the AIS"
        % (L_hillock, L_AIS, diams_hillock, diams_AIS)
    )
