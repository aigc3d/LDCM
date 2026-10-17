# Integrated into LDCM; build against the active PyTorch/CUDA environment.
"""Optional ATen CUDA reductions; the portable graph solver remains a fallback."""
from pathlib import Path
import os
import shutil
import threading

_lock=threading.Lock()
_attempted=False
_module=None
_reason='not initialized'


def get_backend():
    global _attempted,_module,_reason
    with _lock:
        if _attempted:
            return _module
        _attempted=True
        native_flag = os.environ.get('LDCM_POISSON_NATIVE',
            os.environ.get('QUALITY_POISSON_NATIVE', os.environ.get('FAST_POISSON_NATIVE', '1')))
        if native_flag == '0':
            _reason='disabled by LDCM_POISSON_NATIVE=0 (or a legacy Poisson native switch)'
            return None
        try:
            from torch.utils.cpp_extension import CUDA_HOME,load
            if CUDA_HOME is None or not (Path(CUDA_HOME)/'bin/nvcc').is_file() or shutil.which('ninja') is None:
                _reason='CUDA compiler or ninja is unavailable'
                return None
            root=Path(__file__).parent
            _module=load(name='ldcm_poisson_native',
                sources=[str(root/'_native.cpp'),str(root/'_native.cu')],
                extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=False)
            _reason='ATen CUDA residual reductions enabled'
        except (OSError,RuntimeError) as exc:
            _reason=f'native extension unavailable: {type(exc).__name__}: {exc}'
        return _module


def backend_info():
    """Report whether optional native reductions were loaded (without compiling)."""
    return dict(initialized=_attempted,native=_module is not None,reason=_reason)
