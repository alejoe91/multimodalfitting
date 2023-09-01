import pandas
import logging

logger = logging.getLogger(__name__)

from bluepyopt.ephys.recordings import Recording
from bluepyopt.ephys.responses import Response


class CompCurrentRecording(Recording):

    """Response to stimulus"""

    def __init__(
            self,
            name=None,
            location=None,
            variable='v'):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location in the model of the recording
            variable (str): which variable to record from (e.g. 'v')
        """

        super(CompCurrentRecording, self).__init__(
            name=name)
        self.location = location
        self.variable = variable

        self.varvector = None
        self.tvector = None

        self.time = None
        self.current = None

        self.instantiated = False

    @property
    def response(self):
        """Return recording response"""

        if not self.instantiated:
            return None

        return TimeCurrentResponse(self.name,
                                   self.tvector.to_python(),
                                   self.varvector.to_python())

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording"""

        logger.debug('Adding compartment recording of %s at %s',
                     self.variable, self.location)

        self.varvector = sim.neuron.h.Vector()
        seg = self.location.instantiate(sim=sim, icell=icell)
        self.varvector.record(getattr(seg, '_ref_%s' % self.variable))

        self.tvector = sim.neuron.h.Vector()
        self.tvector.record(sim.neuron.h._ref_t)  # pylint: disable=W0212

        self.instantiated = True

    def destroy(self, sim=None):
        """Destroy recording"""

        self.varvector = None
        self.tvector = None
        self.instantiated = False

    def __str__(self):
        """String representation"""

        return '%s: %s at %s' % (self.name, self.variable, self.location)


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
        self.response['time'] = pandas.Series(time)
        self.response['current'] = pandas.Series(current)

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
            self.response['time'],
            self.response['current'],
            label='%s' %
            self.name)