import logging
import math

logger = logging.getLogger(__name__)


def fix_morphology_exp(sim=None, icell=None, nseg_ais=50, abd=False):
    '''
    Fix exp morphology by renaming 'apic' -> 'ais' and by setting the number of ais segments.
    In addition, if 'abd=True' an axon-bearing-dendrite section is created between the soma and the ais.

    Parameters
    ----------
    sim: BlueyuOpt.Simulator
    icell: BlueyuOpt.CellModel
    nseg_ais: int
        Number of segment for AIS
    abd: bool
        If True, the axon-bearing-dendrite section is created for the dendrite
        connecting the soma to the AIS
    '''

    # 1) rename apic to ais
    sim.neuron.h.execute("create ais[1]", icell)
    sec_ais = icell.ais[0]

    for sec in icell.apical:
        # add 3d points
        n3d = sec.n3d()
        for i in range(n3d):
            sim.neuron.h.pt3dadd(sec.x3d(i), sec.y3d(i), sec.z3d(i), sec.diam3d(i), sec=sec_ais)

        # connect parent and children
        parent_ais = sec.parentseg().sec
        children_ais = sec.children()

        parent_x = sim.neuron.h.parent_connection(sec=sec)
        child_y = sim.neuron.h.section_orientation(sec=sec)
        sec_ais.connect(parent_ais, parent_x, child_y)

        for child in children_ais:
            parent_x = sim.neuron.h.parent_connection(sec=child)
            child_y = sim.neuron.h.section_orientation(sec=child)
            sim.neuron.h.disconnect(sec=child)
            child.connect(sec_ais, parent_x, child_y)

        # delete original apical section
        sim.neuron.h.pt3dclear(sec=sec)
        sim.neuron.h.disconnect(sec=sec)
        sim.neuron.h.delete_section(sec=sec)

        # this should fix the morphology issue
        sim.neuron.h.pt3dconst(1, sec=sec_ais)

        # add ais to section lists
        icell.axon_initial_segment.append(sec=sec_ais)
        icell.all.append(sec=sec_ais)

    # set number of segments
    for index, section in enumerate(icell.axon_initial_segment):
        section.nseg = nseg_ais

    logger.debug(f"Replace apic sections with AIS section with {nseg_ais} segments")

    # add ABD
    if abd:
        ais_sec = icell.ais[0]
        abd_sections = []
        sec = ais_sec

        # count number of sections between AIS and soma
        while "soma" not in sec.name():
            parent_sec = sec.parentseg().sec
            if "soma" not in parent_sec.name():
                abd_sections.append(parent_sec)
            sec = parent_sec

        sim.neuron.h.execute(f"create abd[{len(abd_sections)}]", icell)

        logger.debug(f"Adding {len(abd_sections)} ABD sections")

        for i, sec in enumerate(abd_sections[::-1]):
            sec_abd = icell.abd[i]
            logger.debug(f"{sec.name()} is now {sec_abd.name()}")

            # add 3d points
            n3d = sec.n3d()
            for i in range(n3d):
                sim.neuron.h.pt3dadd(sec.x3d(i), sec.y3d(i), sec.z3d(i), sec.diam3d(i), sec=sec_abd)

            # connect parent and children
            parent_abd = sec.parentseg().sec
            children_abd = sec.children()
            parent_x = sim.neuron.h.parent_connection(sec=sec)
            child_y = sim.neuron.h.section_orientation(sec=sec)

            sec_abd.connect(parent_abd, parent_x, child_y)
            for child in children_abd:
                parent_x = sim.neuron.h.parent_connection(sec=child)
                child_y = sim.neuron.h.section_orientation(sec=child)
                sim.neuron.h.disconnect(sec=child)
                child.connect(sec_abd, parent_x, child_y)

            # delete original section
            sim.neuron.h.disconnect(sec=sec)
            sim.neuron.h.pt3dclear(sec=sec)
            sim.neuron.h.delete_section(sec=sec)

            # this should fix the morphology issue
            sim.neuron.h.pt3dconst(1, sec=sec_abd)

            # add abd section to section lists
            icell.axon_bearing_dendrite.append(sec=sec_abd)
            icell.all.append(sec=sec_abd)

        logger.debug(
            f"Replace {len(abd_sections)} sections with ABD sections"
        )


def replace_axon_with_hillock_ais(sim=None, icell=None, l_hillock=10, l_ais=40,
                                  l_myelin=1000, d_myelin=0.2, seg_len=5, myelin_nseg=5):
    """Replace axon by an hillock and an AIS while keeping the 3d informations of the sections"""

    ''' In this first part, we will increase the number of 3d informations in all the axonal sections in order to 
    get the most precise information about axonal morphology. This is done by interpolating several time the existing 
    3d informations (x, y, z, and diameter)'''

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

        # this number define the number of interpolations made. The higher it is,
        # the most precise will be the axon 3d definition.
        interpolation_number = 7
        # we create 4 intp lists to store the values corresponding at the middle between each existing values
        for j in range(interpolation_number):
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
            # we create 4 new lists to store the existing values and the values in between to obtain the lists
            # interpolated
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
            list_x3d = list_x3d_new[:]
            list_y3d = list_y3d_new[:]
            list_z3d = list_z3d_new[:]
            list_diam3d = list_diam3d_new[:]

            # we erase the 3d info from the section and replace with the new interpolated 3d info
        sim.neuron.h.pt3dclear(sec=section)
        for i in range(len(list_x3d)):
            sim.neuron.h.pt3dadd(list_x3d[i], list_y3d[i], list_z3d[i], list_diam3d[i], sec=section)

    l_total = l_hillock + l_ais
    final_seg_length = seg_len  # the final length of the segments during the simulation

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
                # this line increases the distance from the soma at each new 3d info point
                dist_from_soma = dist_from_soma + sim.neuron.h.arc3d(i, sec=section) - sim.neuron.h.arc3d(i - 1,
                                                                                                          sec=section)

            if dist_from_soma <= l_hillock:
                x3d_hillock.append(sim.neuron.h.x3d(i, sec=section))
                y3d_hillock.append(sim.neuron.h.y3d(i, sec=section))
                z3d_hillock.append(sim.neuron.h.z3d(i, sec=section))
                diam3d_hillock.append(sim.neuron.h.diam3d(i, sec=section))

            elif dist_from_soma > l_hillock and dist_from_soma <= l_total:
                x3d_ais.append(sim.neuron.h.x3d(i, sec=section))
                y3d_ais.append(sim.neuron.h.y3d(i, sec=section))
                z3d_ais.append(sim.neuron.h.z3d(i, sec=section))
                diam3d_ais.append(sim.neuron.h.diam3d(i, sec=section))

            else:
                break

        if dist_from_soma > l_total:
            break

    '''In this third part, we will delete all the axon sections, create hillock and ais section, 
    add the 3d info to these new sections, and connect them'''

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create hillock[1]", icell)
    sim.neuron.h.execute("create ais[1]", icell)

    icell.hillockal.append(sec=icell.hillock[0])
    icell.all.append(sec=icell.hillock[0])

    icell.axon_initial_segment.append(sec=icell.ais[0])
    icell.all.append(sec=icell.ais[0])

    for i in range(len(x3d_hillock)):
        sim.neuron.h.pt3dadd(x3d_hillock[i], y3d_hillock[i], z3d_hillock[i], diam3d_hillock[i], sec=icell.hillock[0])

    for i in range(len(x3d_ais)):
        sim.neuron.h.pt3dadd(x3d_ais[i], y3d_ais[i], z3d_ais[i], diam3d_ais[i], sec=icell.ais[0])

    icell.hillock[0].nseg = 1 + int(l_hillock / final_seg_length)
    icell.ais[0].nseg = 1 + int(l_ais / final_seg_length)

    # childsec.connect(parentsec, parentx, childx)
    icell.hillock[0].connect(icell.soma[0], 1.0, 0.0)
    icell.ais[0].connect(icell.hillock[0], 1.0, 0.0)

    '''In this fourth part, we will create a myelin section and connect it to the ais section'''

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = myelin_nseg
    icell.myelin[0].L = l_myelin
    icell.myelin[0].diam = d_myelin  # diams[count-1]
    icell.myelin[0].connect(icell.ais[0], 1.0, 0.0)

    diams_hillock = []
    diams_ais = []
    for seg in icell.hillock[0]:
        diams_hillock.append(seg.diam)
    for seg in icell.ais[0]:
        diams_ais.append(seg.diam)

    logger.debug(
        f"Replace axon with hillock of length {l_hillock} and AIS of length {l_ais}, diameters are {diams_hillock} "
        f"for the hillock and {diams_ais} for the AIS"
    )


def replace_axon_with_ais(sim=None, icell=None, l_ais=35, ais_seg_len=5,
                          l_myelin=1000, d_myelin=0.2, myelin_nseg=5):
    """Replace axon by an AIS while keeping the 3d informations of the sections"""

    ''' In this first part, we will increase the number of 3d informations in all the axonal sections in order to 
    get the most precise information about axonal morphology. This is done by interpolating several time the existing 
    3d informations (x, y, z, and diameter)'''

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

        # this number define the number of interpolations made. The higher it is,
        # the most precise will be the axon 3d definition.
        interpolation_number = 7
        # we create 4 intp lists to store the values corresponding at the middle between each existing values
        for j in range(interpolation_number):
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
            # we create 4 new lists to store the existing values and the values in between to obtain the lists
            # interpolated
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
            list_x3d = list_x3d_new[:]
            list_y3d = list_y3d_new[:]
            list_z3d = list_z3d_new[:]
            list_diam3d = list_diam3d_new[:]

            # we erase the 3d info from the section and replace with the new interpolated 3d info
        sim.neuron.h.pt3dclear(sec=section)
        for i in range(len(list_x3d)):
            sim.neuron.h.pt3dadd(list_x3d[i], list_y3d[i], list_z3d[i], list_diam3d[i], sec=section)

    # changed from 35 to make nseg odd. define the desired length of AIS
    # final_seg_length = ais_seg_len  # the final length of the segments during the simulation

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
                # this line increase the distance from the soma at each new 3d info point
                dist_from_soma = dist_from_soma + sim.neuron.h.arc3d(i, sec=section) - sim.neuron.h.arc3d(i - 1,
                                                                                                          sec=section)

            if dist_from_soma <= l_ais:
                x3d_ais.append(sim.neuron.h.x3d(i, sec=section))
                y3d_ais.append(sim.neuron.h.y3d(i, sec=section))
                z3d_ais.append(sim.neuron.h.z3d(i, sec=section))
                diam3d_ais.append(sim.neuron.h.diam3d(i, sec=section))

            else:
                break

        if dist_from_soma > l_ais:
            break

    '''In this third part, we will delete all the axon sections, create hillock and ais section, 
    add the 3d info to these new sections, and connect them'''

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create ais[1]", icell)

    icell.axon_initial_segment.append(sec=icell.ais[0])
    icell.all.append(sec=icell.ais[0])

    for i in range(len(x3d_ais)):
        sim.neuron.h.pt3dadd(x3d_ais[i], y3d_ais[i], z3d_ais[i], diam3d_ais[i], sec=icell.ais[0])

    icell.ais[0].nseg = 1 + int(l_ais / ais_seg_len)

    # childsec.connect(parentsec, parentx, childx)
    icell.ais[0].connect(icell.soma[0], 1.0, 0.0)

    '''In this fourth part, we will create a myelin section and connect it to the ais section'''

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = myelin_nseg
    icell.myelin[0].L = l_myelin
    icell.myelin[0].diam = d_myelin  # diams[count-1]
    icell.myelin[0].connect(icell.ais[0], 1.0, 0.0)

    diams_ais = []
    for seg in icell.ais[0]:
        diams_ais.append(seg.diam)

    logger.debug(
        f"Replace axon with AIS of length {l_ais} and diameter {diams_ais}"
    )
