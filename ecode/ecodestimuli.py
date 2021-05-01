from bluepyopt.ephys.stimuli import Stimulus
from bluepyopt.ephys.recordings import Recording
from bluepyopt.ephys.responses import Response

import LFPy
import numpy as np
import pandas

import logging

logger = logging.getLogger(__name__)

class sAHP(Stimulus):
    """sAHP current clamp injection"""

    def __init__(self,
                 holding_amplitude=0.0,
                 delay=250.0,
                 sahp_tmid=500.0,
                 sahp_tmid2=725.0,
                 sahp_toff=1175.0,
                 total_duration=1425.0,
                 phase1_amplitude=None,
                 phase2_amplitude=None,
                 phase3_amplitude=None,
                 location=None):
        """Constructor

        Args:
            holding_amplitude : holding potential (mV)
            delay (float): time to longstep of sahp (ms)
            sahp_tmid : time to second delay (ms) 
            sahp_tmid2: time to second delay being off (ms) 
            sahp_toff (float): amplitude at end of sahp (nA)
            total_duration (float): total duration (ms)
            phase1_amplitude (float): amplitude of phase1 (nA)
            phase1_amplitude (float): amplitude of phase2 (nA)
            phase1_amplitude (float): amplitude of phase3 (nA)
            location (Location): stimulus Location
        """

        super().__init__()
        self.holding_amplitude = holding_amplitude
        self.delay = delay
        self.sahp_tmid = sahp_tmid
        self.sahp_tmid2 = sahp_tmid2
        self.sahp_toff = sahp_toff
        self.total_duration = total_duration
        self.phase1_amplitude = phase1_amplitude
        self.phase2_amplitude = phase2_amplitude
        self.phase3_amplitude = phase3_amplitude
        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        # create vector to store the times at which stim amp changes
        times = sim.neuron.h.Vector()
        # create vector to store to which stim amps over time
        amps = sim.neuron.h.Vector()

        # at time 0.0, current is holding_amplitude
        times.append(0.0)
        amps.append(self.holding_amplitude)

        # until time delay, current is holding_amplitude
        times.append(self.delay)
        amps.append(self.holding_amplitude)

        times.append(self.delay)
        amps.append(self.holding_amplitude + self.phase1_amplitude)

        times.append(self.sahp_tmid)
        amps.append(self.holding_amplitude + self.phase1_amplitude)

        times.append(self.sahp_tmid)
        amps.append(self.holding_amplitude + self.phase2_amplitude)

        times.append(self.sahp_tmid2)
        amps.append(self.holding_amplitude + self.phase2_amplitude)

        times.append(self.sahp_tmid2)
        amps.append(self.holding_amplitude + self.phase3_amplitude)

        times.append(self.sahp_toff)
        amps.append(self.holding_amplitude + self.phase3_amplitude)

        times.append(self.sahp_toff)
        amps.append(self.holding_amplitude)

        times.append(self.total_duration)
        amps.append(self.holding_amplitude)

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)

        stim = LFPyCell._hoc_stimlist[0]

        # play the above current amplitudes into the current clamp
        amps.play(stim._ref_amp, times, 1)  # pylint: disable=W0212

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""

        # Destroy all persistent objects
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "sAHP phase 1 - %f phase 2  %f phase 3 %f  - delay %f mid delay %f  final delay %f" \
               "totdur %f at %s" % (
                   self.phase1_amplitude,
                   self.phase2_amplitude,
                   self.phase3_amplitude,
                   self.delay,
                   self.sahp_tmid,
                   self.sahp_tmid2,
                   self.total_duration,
                   self.location)


class HyperDepol(Stimulus):
    """HyperDepol Protocol"""

    def __init__(self,
                 holding_amplitude=0.0,
                 hyperpol_amplitude=None,
                 depol_amplitude=None,
                 delay=250.0,
                 tmid=700.0,
                 toff=970.0,
                 total_duration=1220.0,
                 location=None):
        """Constructor

        Args:
            amplitude (float): amplitude (nA)
            location (Location): stimulus Location
        """
        super().__init__()
        self.hyperpol_amplitude = hyperpol_amplitude
        self.depol_amplitude = depol_amplitude
        self.holding_amplitude = holding_amplitude
        self.delay = delay
        self.tmid = tmid
        self.toff = toff
        self.total_duration = total_duration
        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        amps = sim.neuron.h.Vector()
        times = sim.neuron.h.Vector()

        times.append(0.0)
        amps.append(self.holding_amplitude)

        times.append(self.delay)
        amps.append(self.holding_amplitude)

        times.append(self.delay)
        amps.append(self.holding_amplitude + self.hyperpol_amplitude)

        times.append(self.tmid)
        amps.append(self.holding_amplitude + self.hyperpol_amplitude)

        times.append(self.tmid)
        amps.append(self.holding_amplitude + self.depol_amplitude)

        times.append(self.toff)
        amps.append(self.holding_amplitude + self.depol_amplitude)

        times.append(self.toff)
        amps.append(self.holding_amplitude)

        times.append(self.total_duration)
        amps.append(self.holding_amplitude)

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)

        stim = LFPyCell._hoc_stimlist[0]

        amps.play(
            stim._ref_amp,  # pylint:disable=W0212
            times,
            1
        )

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "HyperDepol with hyperopolarization amp %f and depolarization amplitude of %f at %s" % (
            self.hyperpol_amplitude,
            self.depol_amplitude,
            self.location)


class PosCheops(Stimulus):
    """PosCheops Protocol"""

    def __init__(self,
                 delay=250.0,
                 holding_amplitude=0,
                 ramp1_dur=4000.0,
                 ramp2_dur=2000.0,
                 ramp3_dur=1330.0,
                 ramp1_amp=None,
                 ramp2_amp=None,
                 ramp3_amp=None,
                 ramp12_delay=2000.0,
                 ramp23_delay=2000.0,
                 total_duration=20910.0,
                 location=None):
        """Constructor

        Args:
            TODO: CHANGE THESE
            amplitude (float): amplitude (nA)
            location (Location): stimulus Location
        """
        super().__init__()
        self.holding_amplitude = holding_amplitude
        self.delay = delay
        self.holding_amplitude = holding_amplitude
        self.ramp1_dur = ramp1_dur
        self.ramp2_dur = ramp2_dur
        self.ramp3_dur = ramp3_dur
        self.ramp1_amp = ramp1_amp
        self.ramp2_amp = ramp2_amp
        self.ramp3_amp = ramp3_amp
        self.ramp12_delay = ramp12_delay
        self.ramp23_delay = ramp23_delay
        self.total_duration = total_duration
        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        current_vec = sim.neuron.h.Vector()
        time_vec = sim.neuron.h.Vector()

        time_vec.append(0.0)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay + self.ramp1_dur)
        current_vec.append(self.holding_amplitude + self.ramp1_amp)

        time_vec.append(self.delay + 2 * self.ramp1_dur)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay + 2 * self.ramp1_dur + self.ramp12_delay)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay + 2 * self.ramp1_dur + self.ramp12_delay + self.ramp2_dur)
        current_vec.append(self.holding_amplitude + self.ramp2_amp)

        time_vec.append(self.delay + 2 * self.ramp1_dur + self.ramp12_delay + 2 * self.ramp2_dur)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay + 2 * self.ramp1_dur + self.ramp12_delay + 2 * self.ramp2_dur + self.ramp23_delay)
        current_vec.append(self.holding_amplitude)

        time_vec.append(
            self.delay + 2 * self.ramp1_dur + self.ramp12_delay + 2 * self.ramp2_dur + self.ramp23_delay + self.ramp3_dur)
        current_vec.append(self.holding_amplitude + self.ramp3_amp)

        time_vec.append(
            self.delay + 2 * self.ramp1_dur + self.ramp12_delay + 2 * self.ramp2_dur + self.ramp23_delay + 2 * self.ramp3_dur)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.total_duration)
        current_vec.append(self.holding_amplitude)

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)
        stim = LFPyCell._hoc_stimlist[0]

        current_vec.play(
            stim._ref_amp,  # pylint:disable=W0212
            time_vec,
            1
        )

        # Make sure the following objects survive after instantiation
        self.persistent.append(time_vec)
        self.persistent.append(current_vec)

    def destroy(self, sim=None):
        """Destroy stimulus"""
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "PosCheops with ramp1_amp %f - ramp1_dur %f, ramp2_amp %f - ramp2_dur %f, and ramp3_amp %f " \
               "- ramp3_dur %fat %s" % (
            self.ramp1_amp,
            self.ramp1_dur,
            self.ramp2_amp,
            self.ramp2_dur,
            self.ramp3_amp,
            self.ramp3_dur,
            self.location)


class NoiseOU3(Stimulus):
    """NoiseOU3 injection"""

    def __init__(self,
                 filename=None,
                 total_duration=1000.0,
                 location=None):
        """Constructor

        Args:
            filename (string): filepath for Noise
            location (Location): stimulus Location
        """

        super(NoiseOU3, self).__init__()
        self.filename = filename
        self.total_duration = total_duration
        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        noisearray = np.loadtxt(self.filename, delimiter="\t", skiprows=1)

        # create vector to store the times at which stim amp changes
        times = sim.neuron.h.Vector(noisearray[:, 0])
        # create vector to store to which stim amps over time
        amps = sim.neuron.h.Vector(noisearray[:, 1])

        # amps = noisearray[:,1]
        # times = noisearray[:,0]

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)

        stim = LFPyCell._hoc_stimlist[0]

        # play the above current amplitudes into the current clamp
        amps.play(stim._ref_amp, times, 1)  # pylint: disable=W0212

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""

        # Destroy all persistent objects
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "NoiseOU3 from %s located at %s" % (
            self.filename,
            self.location)


#### Stim response and recording ###
class TimeCurrentResponse(Response):

    """Response to stimulus"""

    def __init__(self, name, time=None, current=None):
        """Constructor

        Args:
            name (str): name of this object
            time (list of floats): time series
            current (list of floats): current series
        """

        super(TimeCurrentResponse, self).__init__(name)

        self.response = pandas.DataFrame()
        self.response["time"] = pandas.Series(time)
        self.response["current"] = pandas.Series(current)

    def read_csv(self, filename):
        """Load response from csv file"""

        self.response = pandas.read_csv(filename)

    def to_csv(self, filename):
        """Write response to csv file"""

        self.response.to_csv(filename)

    def __getitem__(self, index):
        """Return item at index"""

        return self.response.__getitem__(index)

    # This plot has to be generalised to several subplots
    def plot(self, axes):
        """Plot the response"""

        axes.plot(
            self.response["time"],
            self.response["current"],
            label="%s" % self.name,
        )


class StimRecording(Recording):

    """Stimulus response"""

    location = "stimulus"
    variable = "i"

    def __init__(self, name=None):
        """Constructor

        Args:
            name (str): name of this object
        """

        super(StimRecording, self).__init__(name=name)

        self.cell = None
        self.tvector = None
        self.time = None

        self.instantiated = False

    @property
    def response(self):
        """Return recording response"""

        if not self.instantiated:
            return None
        self.tvector = self.cell.tvec
        # if len(self.cell.pointprocesses) > 1:
        #     raise Exception
        return TimeCurrentResponse(
            self.name, self.tvector, self.cell.pointprocesses[0].i
        )

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        import LFPy

        """Instantiate recording"""

        logger.debug(
            "Adding recording of %s at %s", self.variable, self.location
        )

        assert isinstance(
            LFPyCell, LFPy.Cell
        ), "LFPRecording is only available for LFPCellModel"
        self.cell = LFPyCell
        self.tvector = None
        self.instantiated = True

    def destroy(self, sim=None):
        """Destroy recording"""
        self.tvector = None
        self.instantiated = False

    def __str__(self):
        """String representation"""

        return "%s: %s at %s" % (self.name, self.variable, self.location)
