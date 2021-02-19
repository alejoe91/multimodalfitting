import logging

logger = logging.getLogger(__name__)


def replace_axon_with_hillock(sim=None, icell=None, l_hillock=10, l_ais=35, l_myelin=1000):
    """Replace axon"""

    """Replace axon by an hillock and an AIS while keeping the 3d informations of the sections"""

    ''' In this first part, we will increase the number of 3d informations in all the axonal sections in order to get the most precise information about axonal morphology. This is done by interpolating several time the existing 3d informations (x, y, z, and diameter)'''

    for index, section in enumerate(icell.axonal):
        # first, we create 4 list to store the 3d informations
        list_x3d = []
        list_y3d = []
        list_z3d = []
        list_diam3d = []
        for i in range(int(sim.neuron.h.n3d(sec=section))):  # then, we store the 3d informations in those lists
            list_x3d.append(sim.neuron.h.x3d(i, sec=section))
            list_y3d.append(sim.neuron.h.y3d(i, sec=section))
            list_z3d.append(sim.neuron.h.z3d(i, sec=section))
            list_diam3d.append(sim.neuron.h.diam3d(i, sec=section))

        interpolation_number = 7  # this number define the number of interpolations made. The higher it is, the most precise will be the axon 3d definition.
        for j in range(
                interpolation_number):  # we create 4 intp lists to store the values corresponding at the middle between each existing values
            list_x3d_intp = []
            list_y3d_intp = []
            list_z3d_intp = []
            list_diam3d_intp = []
            for i in range(len(
                    list_x3d) - 1):  # we store the the values corresponding at the middle between each existing values
                list_x3d_intp.append((list_x3d[i] + list_x3d[i + 1]) / 2)
                list_y3d_intp.append((list_y3d[i] + list_y3d[i + 1]) / 2)
                list_z3d_intp.append((list_z3d[i] + list_z3d[i + 1]) / 2)
                list_diam3d_intp.append((list_diam3d[i] + list_diam3d[i + 1]) / 2)
            # we create 4 new lists to store the existing values and the values in between to obtain the lists interpolated
            list_x3d_new = []
            list_y3d_new = []
            list_z3d_new = []
            list_diam3d_new = []
            for i in range(len(list_x3d) - 1):
                list_x3d_new.append(list_x3d[i])
                list_x3d_new.append(list_x3d_intp[i])
                list_y3d_new.append(list_y3d[i])
                list_y3d_new.append(list_y3d_intp[i])
                list_z3d_new.append(list_z3d[i])
                list_z3d_new.append(list_z3d_intp[i])
                list_diam3d_new.append(list_diam3d[i])
                list_diam3d_new.append(list_diam3d_intp[i])
            list_x3d_new.append(list_x3d[-1])
            list_y3d_new.append(list_y3d[-1])
            list_z3d_new.append(list_z3d[-1])
            list_diam3d_new.append(list_diam3d[-1])

            # we erase the firsts lists and put the new values interpolated in it
            list_x3d = []
            list_y3d = []
            list_z3d = []
            list_diam3d = []
            list_x3d = list_x3d_new[:]
            list_y3d = list_y3d_new[:]
            list_z3d = list_z3d_new[:]
            list_diam3d = list_diam3d_new[:]

            # we erase the 3d info from the section and replace with the new interpolated 3d info
        sim.neuron.h.pt3dclear(sec=section)
        for i in range(len(list_x3d)):
            sim.neuron.h.pt3dadd(list_x3d[i], list_y3d[i], list_z3d[i], list_diam3d[i], sec=section)

    '''In this second part, we will store the 3d info of hillock and AIS into lists'''

    L_hillock = l_hillock  # define the desired hillock length
    L_AIS = l_ais  # define the desired length of AIS
    L_total = L_hillock + L_AIS
    final_seg_length = 5  # the final length of the segments during the simulation

    # lists to append the future 3d info of hillock
    x3d_hillock = []
    y3d_hillock = []
    z3d_hillock = []
    diam3d_hillock = []
    # lists to append the future 3d info of ais
    x3d_ais = []
    y3d_ais = []
    z3d_ais = []
    diam3d_ais = []

    dist_from_soma = 0
    for index, section in enumerate(icell.axonal):

        for i in range(int(sim.neuron.h.n3d(sec=section))):

            if i == 0:
                dist_from_soma = dist_from_soma
            else:
                dist_from_soma = dist_from_soma + sim.neuron.h.arc3d(i, sec=section) - sim.neuron.h.arc3d(i - 1,
                                                                                                          sec=section)  # this line increase the distance from the soma at each new 3d info point

            if dist_from_soma <= L_hillock:
                x3d_hillock.append(sim.neuron.h.x3d(i, sec=section))
                y3d_hillock.append(sim.neuron.h.y3d(i, sec=section))
                z3d_hillock.append(sim.neuron.h.z3d(i, sec=section))
                diam3d_hillock.append(sim.neuron.h.diam3d(i, sec=section))

            elif dist_from_soma > L_hillock and dist_from_soma <= L_total:
                x3d_ais.append(sim.neuron.h.x3d(i, sec=section))
                y3d_ais.append(sim.neuron.h.y3d(i, sec=section))
                z3d_ais.append(sim.neuron.h.z3d(i, sec=section))
                diam3d_ais.append(sim.neuron.h.diam3d(i, sec=section))

            else:
                break

        if dist_from_soma > L_total:
            break

    '''In this third part, we will delete all the axon sections, create hillock and ais section, add the 3d info to these new sections, and connect them'''

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

        #  new axon array
    sim.neuron.h.execute("create hillock", icell)
    sim.neuron.h.execute("create ais", icell)

    icell.hillockal.append(sec=icell.hillock)
    icell.all.append(sec=icell.hillock)

    icell.axon_initial_segment.append(sec=icell.ais)
    icell.all.append(sec=icell.ais)

    for i in range(len(x3d_hillock)):
        sim.neuron.h.pt3dadd(x3d_hillock[i], y3d_hillock[i], z3d_hillock[i], diam3d_hillock[i], sec=icell.hillock)

    for i in range(len(x3d_ais)):
        sim.neuron.h.pt3dadd(x3d_ais[i], y3d_ais[i], z3d_ais[i], diam3d_ais[i], sec=icell.ais)

    icell.hillock.nseg = 1 + int(L_hillock / final_seg_length)
    icell.ais.nseg = 1 + int(L_AIS / final_seg_length)

    # childsec.connect(parentsec, parentx, childx)
    icell.hillock.connect(icell.soma[0], 1.0, 0.0)
    icell.ais.connect(icell.hillock, 1.0, 0.0)

    '''In this fourth part, we will create a myelin section and connect it to the ais section'''

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = l_myelin
    # TODO use myelin direction
    icell.myelin[0].diam = diam3d_ais[-1]
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
