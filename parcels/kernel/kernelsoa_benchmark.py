from ctypes import byref
from ctypes import c_double
from ctypes import c_int
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
from parcels.tools.loggers import logger

from parcels.kernel import KernelSOA
from parcels.tools.performance_logger import TimingLog

__all__ = ['KernelSOA_Benchmark']


class KernelSOA_Benchmark(KernelSOA):
    """Kernel object that encapsulates auto-generated code.

    :arg fieldset: FieldSet object providing the field information
    :arg ptype: PType object for the kernel particle
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None,
                 funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super(KernelSOA_Benchmark, self).__init__(fieldset, ptype, pyfunc, funcname, funccode, py_ast, funcvars, c_include, delete_cfiles)
        self._compute_timings = TimingLog()
        self._io_timings = TimingLog()
        self._mem_io_timings = TimingLog()

    @property
    def io_timings(self):
        return self._io_timings

    @property
    def mem_io_timings(self):
        return self._mem_io_timings

    @property
    def compute_timings(self):
        return self._compute_timings

    def __del__(self):
        super(KernelSOA_Benchmark, self).__del__()

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        self._io_timings.start_timing()
        self.load_fieldset_jit(pset)
        self._io_timings.stop_timing()
        self._io_timings.accumulate_timing()

        self._compute_timings.start_timing()
        fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
        fargs += [c_double(f) for f in self.const_args.values()]
        particle_data = byref(pset.ctypes_struct)
        result = self._function(c_int(len(pset)),
                                particle_data,
                                c_double(endtime),
                                c_double(dt),
                                *fargs)
        self._compute_timings.stop_timing()
        self._compute_timings.accumulate_timing()

        self._io_timings.advance_iteration()
        self._mem_io_timings.advance_iteration()
        self._compute_timings.advance_iteration()
        return result

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        # sign of dt: { [0, 1]: forward simulation; -1: backward simulation }
        sign_dt = np.sign(dt)

        analytical = False
        if 'AdvectionAnalytical' in self._pyfunc.__name__:
            analytical = True
            if not np.isinf(dt):
                logger.warning_once('dt is not used in AnalyticalAdvection, so is set to np.inf')
            dt = np.inf

        if self.fieldset is not None:
            for f in self.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                self._io_timings.start_timing()
                loaded_data = f.data
                self._io_timings.stop_timing()
                self._io_timings.accumulate_timing()
                self._mem_io_timings.start_timing()
                f.data = np.array(loaded_data)
                self._mem_io_timings.stop_timing()
                self._mem_io_timings.accumulate_timing()

        self._compute_timings.start_timing()
        for p in pset:
            self.evaluate_particle(p, endtime, sign_dt, dt, analytical=analytical)
        self._compute_timings.stop_timing()
        self._compute_timings.accumulate_timing()

        self._io_timings.advance_iteration()
        self._mem_io_timings.advance_iteration()
        self._compute_timings.advance_iteration()

    def remove_deleted(self, pset, output_file, endtime):
        """Utility to remove all particles that signalled deletion"""
        self._mem_io_timings.start_timing()
        super(KernelSOA_Benchmark, self).remove_deleted(pset=pset, output_file=output_file, endtime=endtime)
        self._mem_io_timings.stop_timing()
        self._mem_io_timings.accumulate_timing()
        self._mem_io_timings.advance_iteration()

    def __add__(self, kernel):
        if not isinstance(kernel, KernelSOA_Benchmark):
            kernel = KernelSOA_Benchmark(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, KernelSOA_Benchmark)

    def __radd__(self, kernel):
        if not isinstance(kernel, KernelSOA_Benchmark):
            kernel = KernelSOA_Benchmark(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, KernelSOA_Benchmark)
