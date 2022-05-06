r"""
==========================================================
Pretty-printer for quantum objects (:mod:`rqutils.qprint`)
==========================================================

.. currentmodule:: rqutils.qprint

QPrint API
==========

.. autosummary::
   :toctree: ../generated
   
   qprint
   pauliprint
"""

from typing import Tuple, List, Sequence, Union, Optional, Any
from numbers import Number
import builtins
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import scipy
except ImportError:
    has_scipy = False
else:
    has_scipy = True

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    has_mpl = False
else:
    has_mpl = True

    MATPLOTLIB_INLINE_BACKENDS = {
        "module://ipykernel.pylab.backend_inline",
        "module://matplotlib_inline.backend_inline",
        "nbAgg",
    }
    
try:
    from qutip import Qobj
except ImportError:
    has_qutip = False
else:
    has_qutip = True
    
from . import paulis
from ._types import array_like

class LaTeXRepr:
    def __init__(self, obj):
        self.obj = obj
        
    def _repr_latex_(self):
        return self.obj.latex()

PrintReturnType = Union[str, LaTeXRepr]
if has_mpl:
    PrintReturnType = Union[PrintReturnType, mpl.figure.Figure]

def qprint(
    qobj: Any,
    fmt: str = 'braket',
    amp_norm: Optional[Union[Number, Tuple[Number, str]]] = None,
    phase_norm: Optional[Tuple[Number, str]] = (np.pi, 'π'),
    global_phase: Optional[Union[Number, str]] = None,
    terms_per_row: int = 0,
    amp_format: str = '.3f',
    phase_format: str = '.2f',
    epsilon: float = 1.e-6,
    lhs_label: Optional[str] = None,
    dim: Optional[array_like] = None,
    binary: bool = False,
    symbol: Optional[Union[str, Sequence[str]]] = None,
    delimiter: str = '',
    output: Optional[str] = None
) -> PrintReturnType:
    """Pretty-print a quantum object.
    
    Available output formats are
    - `'braket'`: For a column vector, row vector, or matrix input. Prints out the mathematical
      expression of the linear combination of bras, kets, or ket-bras.
    - `'pauli'`: For an input representing a square matrix (shape `(d1*d2*..., d1*d2*...)`) or a
      components array (shape `(d1**2, d2**2, ...)`). Argument `dim` is required for the matrix
      interpretation.

    Three printing formats are supported:
    - `'text'`: Print text to stdout.
    - `'latex'`: Return an object processable by IPython into a typeset LaTeX expression.
    - `'mpl'`: Return a matplotlib figure.
    
    Args:
        qobj: Input quantum object.
        fmt: Print format (`'braket'` or `'pauli'`).
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        epsilon: Numerical cutoff for ignoring amplitudes (relative to max) and phase (absolute).
        lhs_label: If not None, prepend 'label = ' to the printout.
        dim: Specification of the dimensions of the subsystems. For `fmt='pauli'`, used only when
            `qobj` is a square matrix or a 1D array.
        binary: Show bra and ket indices in binary. Only for `fmt='braket'`.
        symbol: Pauli matrix symbols. Only for `fmt='pauli'`.
        delimiter: Pauli product delimiter. Only for `fmt='pauli'`.
        output: Output method (`'text'`, `'latex'`, or `'mpl'`).
    """
    if output is None:
        if hasattr(builtins, '__IPYTHON__'):
            output = 'latex'
        else:
            output = 'text'

    if fmt == 'braket':
        pobj = QPrintBraKet(qobj=qobj,
                            amp_norm=amp_norm,
                            phase_norm=phase_norm,
                            global_phase=global_phase,
                            terms_per_row=terms_per_row,
                            amp_format=amp_format,
                            phase_format=phase_format,
                            epsilon=epsilon,
                            lhs_label=lhs_label,
                            dim=dim,
                            binary=binary)
        
    elif fmt == 'pauli':
        pobj = QPrintPauli(qobj=qobj,
                           amp_norm=amp_norm,
                           phase_norm=phase_norm,
                           global_phase=global_phase,
                           terms_per_row=terms_per_row,
                           amp_format=amp_format,
                           phase_format=phase_format,
                           epsilon=epsilon,
                           lhs_label=lhs_label,
                           dim=dim,
                           symbol=symbol,
                           delimiter=delimiter)
        
    else:
        raise NotImplementedError(f'qprint with format {fmt} not implemented')
            
    if output == 'text':
        return pobj
    elif output == 'latex':
        return LaTeXRepr(pobj)
    elif output == 'mpl':
        return pobj.mpl()
    else:
        raise NotImplementedError(f'qprint with output {output} not implemented')
        

class QPrintBase:
    """Helper class to compose an expression of a given quantum object.
    
    This is a base class for QPrint which performs numerical processing of the components in the input
    quantum object. Basis labeling is handled by the concrete subclasses.
    
    Args:
        qobj: Input quantum object.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        epsilon: Numerical cutoff for ignoring amplitudes (relative to max) and phase (absolute).
        lhs_label: If not None, prepend 'label = ' to the printout.
    """
    
    @dataclass
    class Term:
        index: tuple
        sign: int
        amp: str
        phase: str
        label: str = ''
        
    def __init__(
        self,
        qobj: Any,
        amp_norm: Optional[Union[Number, Tuple[Number, str]]] = None,
        phase_norm: Optional[Tuple[Number, str]] = (np.pi, 'π'),
        global_phase: Optional[Union[Number, str]] = None,
        terms_per_row: int = 0,
        amp_format: str = '.3f',
        phase_format: str = '.2f',
        epsilon: float = 1.e-6,
        lhs_label: Optional[str] = None
    ):
        self.qobj = qobj
        self.amp_norm = amp_norm
        self.phase_norm = phase_norm
        self.global_phase = global_phase
        self.terms_per_row = terms_per_row
        self.amp_format = amp_format
        self.phase_format = phase_format
        self.epsilon = epsilon
        self.lhs_label = lhs_label
        
    def _process(self, obj=None) -> Tuple[int, str, str, List[List[Term]]]:
        """Compose a list of QPrintTerms."""
        if obj is None:
            obj = self.qobj
            
        qobj, data = self._qobj_data(obj)

        # Amplitude format template
        amp_template = f'{{:{self.amp_format}}}'

        # Phase format template
        phase_template = f'{{:{self.phase_format}}}'

        ## Preprocess the qobj

        # Absolute value and phase of the amplitudes
        absamp = np.abs(data)
        phase = np.angle(data)

        # Normalize the abs amplitudes and identify integral values
        if self.amp_norm is not None:
            if isinstance(self.amp_norm, tuple):
                absamp /= self.amp_norm[0]
                global_amp = self.amp_norm[1]
            else:
                absamp /= self.amp_norm
                if np.isclose(np.round(self.amp_norm), self.amp_norm, rtol=self.epsilon):
                    global_amp = f'{np.round(self.amp_norm)}'
                else:
                    global_amp = amp_template.format(self.amp_norm)
                
        else:
            global_amp = ''
            
        rounded_amp = np.round(absamp).astype(int)
        amp_is_int = np.isclose(rounded_amp, absamp, rtol=self.epsilon)
        rounded_amp = np.where(amp_is_int, rounded_amp, -1)
        
        # Shift and normalize the phases and identify integral values
        phase_offset = 0.
        if self.global_phase is not None:
            if self.global_phase == 'mean':
                phase_offset = np.mean(phase)
            else:
                phase_offset = self.global_phase
                
            phase -= phase_offset

        twopi = 2. * np.pi
        
        while np.any((phase < 0.) | (phase >= twopi)):
            phase = np.where(phase >= 0., phase, phase + twopi)
            phase = np.where(phase < twopi, phase, phase - twopi)

        def normalize_phase(phase):
            reduced_phase = phase / (np.pi / 2.)
            axis_proj = np.round(reduced_phase).astype(int)
            on_axis = np.isclose(axis_proj, reduced_phase, rtol=0., atol=self.epsilon)
            axis_proj = np.where(on_axis, axis_proj, -1)

            if self.phase_norm is not None:
                phase /= self.phase_norm[0]

            rounded_phase = np.round(phase).astype(int)
            phase_is_int = np.isclose(rounded_phase, phase, rtol=0., atol=self.epsilon)
            rounded_phase = np.where(phase_is_int, rounded_phase, -1)
            
            return phase, axis_proj, rounded_phase
        
        def sign_and_phase(phase, axis_proj, rounded_phase):
            if axis_proj == -1:
                # Not on Re or Im axis
                if rounded_phase == -1:
                    expr = phase_template.format(phase)
                else:
                    expr = f'{rounded_phase}'
                    
                sign = 1

            else:
                if axis_proj % 2 == 1:
                    expr = '/'
                else:
                    expr = '0'

                if axis_proj >= 2:
                    sign = -1
                else:
                    sign = 1
                    
            return sign, expr
        
        norm_offset, offset_proj, rounded_offset = normalize_phase(phase_offset)
        global_sign, global_phase = sign_and_phase(norm_offset, offset_proj, rounded_offset)
        
        norm_phase, axis_proj, rounded_phase = normalize_phase(phase)

        ## Compose the terms
        
        # Show only terms with absamp < absamp * epsilon
        amp_atol = np.amax(absamp) * self.epsilon
        amp_is_zero = np.isclose(np.zeros_like(absamp), absamp, atol=amp_atol)
        term_indices = list(zip(*np.logical_not(amp_is_zero).nonzero())) # convert into list of tuples

        # List of terms
        terms = []

        for idx in term_indices:
            sign, phase_expr = sign_and_phase(norm_phase[idx], axis_proj[idx], rounded_phase[idx])
                
            if rounded_amp[idx] == -1:
                amp_expr = amp_template.format(absamp[idx])
            else:
                amp_expr = f'{rounded_amp[idx]}'
                
            terms.append(QPrintBase.Term(index=idx, sign=sign, amp=amp_expr, phase=phase_expr))

        return global_sign, global_amp, global_phase, terms
    
    def _qobj_data(self, obj=None):
        if obj is None:
            obj = self.qobj

        if has_qutip and isinstance(obj, Qobj):
            qobj = obj.data
            data = qobj.data
        elif has_scipy and isinstance(obj, scipy.sparse.csr_matrix):
            qobj = obj
            data = qobj.data
        elif isinstance(obj, np.ndarray):
            qobj = obj
            data = qobj
        else:
            raise NotImplementedError(f'qprint not implemented for {type(obj)}')
            
        return qobj, data
    
    def __repr__(self):
        expr = self._format_lhs('text')
        
        if expr is None:
            expr = ''
        else:
            expr += ' = '

        pre_expr, lines = self._make_lines('text')
        
        if pre_expr:
            expr += f'{pre_expr} ('

        expr += '\n'.join(lines)
        
        if pre_expr:
            expr += ')'
            
        return expr
    
    def latex(self) -> str:
        pre_expr, lines = self._make_lines('latex')

        if pre_expr:
            lines[0] = fr' \left( {lines[0]}'
            lines[-1] += r' \right)'

            if len(lines) > 1:
                lines[0] += r' \right.'
                lines[-1] = r'\left. ' + lines[-1]
                
        if len(lines) > 1:
            lines = list(f'& {line}' for line in lines)
            
        if pre_expr:
            lines[0] = f'{pre_expr} {lines[0]}'
            
        lhs = self._format_lhs('latex')
                
        if lhs is not None:
            lines[0] = f'{lhs} = {lines[0]}'
                
        if len(lines) == 1:
            return fr'$\displaystyle {lines[0]}$'
        else:
            return r'\begin{split} ' + r' \\ '.join(lines) + r' \end{split}'
        
    def mpl(self):
        if not has_mpl:
            raise RuntimeError('Matplotlib is not available')
        
        pre_expr, lines = self._make_lines('latex')
        
        if pre_expr:
            lines[0] = f'{pre_expr} ({lines[0]}'
            lines[-1] += ')'
            
        lhs = self._format_lhs('latex')
            
        if lhs is not None:
            lines[0] = f'{lhs} = {lines[0]}'
            
        fig, ax = plt.subplots(1, figsize=[10., 0.5 * len(lines)])
        ax.axis('off')
    
        num_rows = len(lines)
        for irow, line in enumerate(lines):
            ax.text(0.5, 1. / num_rows * (num_rows - irow - 1), f'${line}$', fontsize='x-large', ha='right')
            
        if mpl.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
            plt.close(fig)
                
        return fig
    
    def _make_lines(self, mode) -> list:
        global_sign, global_amp, global_phase, terms = self._process()
        self._add_labels(terms, mode)
        
        pre_expr = ''
        
        if global_sign == -1:
            pre_expr += '-'
        
        pre_expr += global_amp
        pre_expr += self._format_phase(global_phase, mode)

        lines = []
        line_expr = ''
        num_terms = 0
        
        for term in terms:
            if lines or line_expr:
                if term.sign == -1:
                    line_expr += ' - '
                else:
                    line_expr += ' + '
            elif term.sign == -1:
                line_expr += '-'
                
            if term.amp != '1':
                line_expr += term.amp
                
            line_expr += self._format_phase(term.phase, mode)
            line_expr += term.label
                
            num_terms += 1
            if num_terms == self.terms_per_row:
                lines.append(line_expr)
                line_expr = ''
                num_terms = 0
                
        if num_terms != 0:
            lines.append(line_expr)
            
        if not lines:
            lines = ['0']

        return pre_expr, lines
        
    def _format_phase(self, phase_expr, mode):
        if phase_expr == '0':
            return ''
        elif phase_expr == '/':
            return 'i'
        
        if mode == 'text':
            expr = '['

            if self.phase_norm is not None and self.phase_norm[1]:
                if phase_expr == '1':
                    expr += self.phase_norm[1]
                elif self.phase_norm[1][0].isnumeric():
                    expr += f'{phase_expr}({self.phase_norm[1]})'
                else:
                    expr += f'{phase_expr}{self.phase_norm[1]}'
            else:
                expr += phase_expr

            expr += ']'

        elif mode == 'latex':
            expr = 'e^{'

            if phase_expr != '1':
                expr += phase_expr

            if self.phase_norm is not None:
                if self.phase_norm[1] and self.phase_norm[1][0].isnumeric():
                    expr += r' \cdot '

                expr += self.phase_norm[1]

            expr += ' i}'
        
        return expr


class QPrintBraKet(QPrintBase):
    """Helper class to compose an expression of a given quantum object.

    Args:
        qobj: Input quantum object.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        epsilon: Numerical cutoff for ignoring amplitudes (relative to max) and phase (absolute).
        lhs_label: If not None, prepend 'label = ' to the printout.
        dim: Specification of the dimensions of the subsystems.
        binary: Show bra and ket indices in binary.
    """
    class QobjType(Enum):
        KET = 1
        BRA = 2
        OPER = 3
        
    @staticmethod
    def _type_and_dim(qobj):
        if len(qobj.shape) == 1 or qobj.shape[1] == 1:
            objtype = QPrintBraKet.QobjType.KET
            dim = qobj.shape[0]
        elif qobj.shape[0] == 1 and qobj.shape[1] != 1:
            objtype = QPrintBraKet.QobjType.BRA
            dim = qobj.shape[1]
        else:
            objtype = QPrintBraKet.QobjType.OPER
            dim = qobj.shape[0]
            
        return objtype, dim
    
    def __init__(
        self,
        qobj: Any,
        amp_norm: Optional[Union[Number, Tuple[Number, str]]] = None,
        phase_norm: Optional[Tuple[Number, str]] = (np.pi, 'π'),
        global_phase: Optional[Union[Number, str]] = None,
        terms_per_row: int = 0,
        amp_format: str = '.3f',
        phase_format: str = '.2f',
        epsilon: float = 1.e-6,
        lhs_label: Optional[str] = None,
        dim: Optional[array_like] = None,
        binary: bool = False
    ):
        super().__init__(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            terms_per_row=terms_per_row,
            amp_format=amp_format,
            phase_format=phase_format,
            epsilon=epsilon,
            lhs_label=lhs_label)

        self.dim = dim
        self.binary = binary
        
    def _add_labels(self, terms, mode):
        qobj, _ = self._qobj_data()
        
        objtype, full_dim = self._type_and_dim(qobj)
        
        has_ket = objtype in (QPrintBraKet.QobjType.KET, QPrintBraKet.QobjType.OPER)
        has_bra = objtype in (QPrintBraKet.QobjType.BRA, QPrintBraKet.QobjType.OPER)
        
        dim = self.dim
        if dim is None:
            dim = np.array([full_dim])
            
        assert np.prod(dim) == full_dim, (f'Product of subsystem dimensions {np.prod(dim)}'
                                                f' and qobj dimension {full_dim} do not match')
        
        # State label format template
        if self.binary:
            log2_dims = np.log2(np.asarray(dim))
            assert np.allclose(log2_dims, np.round(log2_dims))
            label_template = ','.join(f'{{:0{s}b}}' for s in log2_dims.astype(int))
        else:
            label_template = ','.join(['{}'] * len(dim))

        # Make tuples of quantum state labels and format the term indices
        if has_scipy and isinstance(qobj, scipy.sparse.csr_matrix):
            # CSR matrix: diff if indptr = number of elements in each row
            repeats = np.diff(qobj.indptr)
            row_labels_flat = np.repeat(np.arange(qobj.shape[0]), repeats)
            # unravel into row indices accounting for the tensor product
            if has_ket:
                row_labels = np.unravel_index(row_labels_flat, dim)
            if has_bra:
                col_labels = np.unravel_index(qobj.indices, dim)
                
        elif isinstance(qobj, np.ndarray):
            if has_ket:
                row_labels = np.unravel_index(np.arange(full_dim), dim)
            if has_bra:
                col_labels = np.unravel_index(np.arange(full_dim), dim)

        # Update the term objects with the basis labels
        for term in terms:
            if has_ket:
                ket_label = label_template.format(*(r[term.index[0]] for r in row_labels))

                if mode == 'text':
                    term.label += f'|{ket_label}>'
                elif mode == 'latex':
                    term.label += fr'| {ket_label} \rangle'
                    
            if has_bra:
                # idx can be an 1- or 2-tuple depending on the type of self.qobj
                bra_label = label_template.format(*(c[term.index[-1]] for c in col_labels))

                if mode == 'text':
                    term.label += f'<{bra_label}|'
                elif mode == 'latex':
                    term.label += fr'\langle {bra_label} |'
                
    def _format_lhs(self, mode) -> Union[str, None]:
        if self.lhs_label:
            objtype, _ = self._type_and_dim(self._qobj_data()[0])

            if mode == 'text':
                if objtype == QPrintBraKet.QobjType.KET:
                    return f'|{self.lhs_label}>'
                elif objtype == QPrintBraKet.QobjType.BRA:
                    return f'<{self.lhs_label}|'

            elif mode == 'latex':
                if objtype == QPrintBraKet.QobjType.KET:
                    return fr'| {self.lhs_label} \rangle'
                elif objtype == QPrintBraKet.QobjType.BRA:
                    return fr'\langle {self.lhs_label} |'

        return self.lhs_label

        
class QPrintPauli(QPrintBase):
    """Helper class to compose an expression for a Pauli decomposition from a matrix or components.

    Args:
        qobj: A square matrix (shape `(d1*d2*..., d1*d2*...)`) or a components array
            (shape `(d1**2, d2**2, ...)`). Argument `dim` is required for the matrix interpretation.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        epsilon: Numerical cutoff for ignoring amplitudes (relative to max) and phase (absolute).
        lhs_label: If not None, prepend 'label = ' to the printout.
        dim: Specification of the dimensions of the subsystems. Used only when `qobj` is a square
            matrix or a 1D array.
        symbol: Pauli matrix symbols.
        delimiter: Pauli product delimiter.
    """
    def __init__(
        self,
        qobj: Any,
        amp_norm: Optional[Union[Number, Tuple[Number, str]]] = None,
        phase_norm: Optional[Tuple[Number, str]] = (np.pi, 'π'),
        global_phase: Optional[Union[Number, str]] = None,
        terms_per_row: int = 0,
        amp_format: str = '.3f',
        phase_format: str = '.2f',
        epsilon: float = 1.e-6,
        lhs_label: Optional[str] = None,
        dim: Optional[array_like] = None,
        symbol: Optional[Union[str, Sequence[str]]] = None,
        delimiter: str = ''
    ):
        super().__init__(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            terms_per_row=terms_per_row,
            amp_format=amp_format,
            phase_format=phase_format,
            epsilon=epsilon,
            lhs_label=lhs_label)

        self.dim = dim
        self.symbol = symbol
        self.delimiter = delimiter
        
    def _process(self):
        qobj, _ = self._qobj_data(self.qobj)
        
        if self.dim is not None:
            if len(qobj.shape) == 2 and qobj.shape[0] == qobj.shape[1]:
                # This is a matrix -> extract the components
                try:
                    matrix = qobj.toarray()
                except AttributeError:
                    matrix = qobj

                qobj = paulis.components(matrix, dim=self.dim)

            elif len(qobj.shape) == 1:
                try:
                    components = qobj.toarray()
                except AttributeError:
                    components = qobj
                    
                qobj = components.reshape(np.square(self.dim))
                
        return super()._process(qobj)
        
    def _add_labels(self, terms, mode):
        qobj, _ = self._qobj_data()
        
        dim = np.around(np.sqrt(qobj.shape)).astype(int)
        
        labels = paulis.labels(dim, symbol=self.symbol, delimiter=self.delimiter,
                               fmt=mode)

        # Update the term objects with the basis labels
        for term in terms:
            if mode == 'text':
                term.label = f'*{labels[term.index]}'
            else:
                term.label = labels[term.index]
    
    def _format_lhs(self, mode) -> Union[str, None]:
        return self.lhs_label
