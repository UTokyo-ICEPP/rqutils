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
from ._types import array_like, MatrixDimension

class LaTeXRepr:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return self.expr

    def __repr__(self):
        return f'LaTeXRepr("{self.expr}")'

    def _repr_latex_(self):
        return self.expr

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

    Returns:
        Object to be printed.
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
        return LaTeXRepr(pobj.latex())
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
        dim: Specification of the dimensions of the subsystems.
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
        lhs_label: Optional[str] = None,
        dim: Optional[MatrixDimension] = None
    ):
        self.amp_norm = amp_norm
        self.phase_norm = phase_norm
        self.global_phase = global_phase
        self.terms_per_row = terms_per_row
        self.amp_format = amp_format
        self.phase_format = phase_format
        self.epsilon = epsilon
        self.lhs_label = lhs_label

        if isinstance(dim, int):
            self._dim = (dim,)
        else:
            self._dim = dim

        self._qobj, self._data = self._qobj_data(qobj)

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

    def latex(self, env='split') -> str:
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

        expr = r' \\ '.join(lines)

        if env:
            return fr'\begin{{{env}}} {expr} \end{{{env}}}'
        else:
            return expr

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

    def _process(self) -> Tuple[int, str, str, List[List[Term]]]:
        """Compose a list of QPrintTerms."""
        # Amplitude format template
        amp_template = f'{{:{self.amp_format}}}'

        # Phase format template
        phase_template = f'{{:{self.phase_format}}}'

        ## Preprocess self._data

        # Absolute value and phase of the amplitudes
        absamp = np.abs(self._data)
        phase = np.angle(self._data)

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

    def _qobj_data(self, qobj):
        if has_qutip and isinstance(qobj, Qobj):
            if self._dim is None:
                self._dim = tuple(qobj.dims[0])

            qobj = qobj.data
            data = qobj.data
        elif has_scipy and isinstance(qobj, scipy.sparse.csr_matrix):
            qobj = qobj
            data = qobj.data
        elif isinstance(qobj, np.ndarray):
            qobj = qobj
            data = qobj
        else:
            raise NotImplementedError(f'qprint not implemented for {type(qobj)}')

        return qobj, data

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
        binary: Show bra and ket indices in binary.
    """
    class QobjType(Enum):
        KET = 1
        BRA = 2
        OPER = 3

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
        dim: Optional[MatrixDimension] = None,
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
            lhs_label=lhs_label,
            dim=dim)

        self.binary = binary

        if len(self._qobj.shape) == 1 or self._qobj.shape[1] == 1:
            self._objtype = QPrintBraKet.QobjType.KET
            self._objdim = self._qobj.shape[0]
        elif self._qobj.shape[0] == 1 and self._qobj.shape[1] != 1:
            self._objtype = QPrintBraKet.QobjType.BRA
            self._objdim = self._qobj.shape[1]
        else:
            self._objtype = QPrintBraKet.QobjType.OPER
            self._objdim = self._qobj.shape[0]

        if self._dim is None:
            self._dim = (self._objdim,)

        if np.prod(self._dim) != self._objdim:
            raise ValueError(f'Product of subsystem dimensions {np.prod(self._dim)} and qobj dimension'
                             f'{self._objdim} do not match')

    def _add_labels(self, terms, mode):
        has_ket = self._objtype in (QPrintBraKet.QobjType.KET, QPrintBraKet.QobjType.OPER)
        has_bra = self._objtype in (QPrintBraKet.QobjType.BRA, QPrintBraKet.QobjType.OPER)

        # State label format template
        if self.binary:
            log2_dims = np.log2(np.asarray(self._dim))
            if not np.allclose(log2_dims, np.round(log2_dims)):
                raise ValueError('Binary labels requested for dimensions not power-of-two')

            label_template = ','.join(f'{{:0{s}b}}' for s in log2_dims.astype(int))
        else:
            label_template = ','.join(['{}'] * len(self._dim))

        # Make tuples of quantum state labels and format the term indices
        if has_scipy and isinstance(self._qobj, scipy.sparse.csr_matrix):
            # CSR matrix: diff if indptr = number of elements in each row
            repeats = np.diff(self._qobj.indptr)
            row_labels_flat = np.repeat(np.arange(self._qobj.shape[0]), repeats)
            # unravel into row indices accounting for the tensor product
            if has_ket:
                row_labels = np.unravel_index(row_labels_flat, self._dim)
            if has_bra:
                col_labels = np.unravel_index(self._qobj.indices, self._dim)

        elif isinstance(self._qobj, np.ndarray):
            if has_ket:
                row_labels = np.unravel_index(np.arange(self._objdim), self._dim)
            if has_bra:
                col_labels = np.unravel_index(np.arange(self._objdim), self._dim)

        # Update the term objects with the basis labels
        for term in terms:
            if has_ket:
                ket_label = label_template.format(*(r[term.index[0]] for r in row_labels))

                if mode == 'text':
                    term.label += f'|{ket_label}>'
                elif mode == 'latex':
                    term.label += fr'| {ket_label} \rangle'

            if has_bra:
                # idx can be an 1- or 2-tuple depending on the type of self._qobj
                bra_label = label_template.format(*(c[term.index[-1]] for c in col_labels))

                if mode == 'text':
                    term.label += f'<{bra_label}|'
                elif mode == 'latex':
                    term.label += fr'\langle {bra_label} |'

    def _format_lhs(self, mode) -> Union[str, None]:
        if self.lhs_label:
            if mode == 'text':
                if self._objtype == QPrintBraKet.QobjType.KET:
                    return f'|{self.lhs_label}>'
                elif self._objtype == QPrintBraKet.QobjType.BRA:
                    return f'<{self.lhs_label}|'

            elif mode == 'latex':
                if self._objtype == QPrintBraKet.QobjType.KET:
                    return fr'| {self.lhs_label} \rangle'
                elif self._objtype == QPrintBraKet.QobjType.BRA:
                    return fr'\langle {self.lhs_label} |'

        return self.lhs_label


class QPrintPauli(QPrintBase):
    """Helper class to compose an expression for a Pauli decomposition from a matrix or components.

    Args:
        qobj: A square matrix (shape `(d1*d2*..., d1*d2*...)`), a structured components array
            (shape `(d1**2, d2**2, ...)`), or a fully flattened components array. Argument `dim` is
            required in the first and third cases.
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
        dim: Optional[MatrixDimension] = None,
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
            lhs_label=lhs_label,
            dim=dim)

        self.symbol = symbol
        self.delimiter = delimiter

    def _qobj_data(self, qobj):
        # Convert all qobj to a components array (shape (d1**2, d2**2, ...))

        qobj, data = super()._qobj_data(qobj)

        if self._dim is not None:
            if len(qobj.shape) == 2 and qobj.shape[0] == qobj.shape[1]:
                # This is a matrix -> extract the components
                try:
                    matrix = qobj.toarray()
                except AttributeError:
                    matrix = qobj

                qobj = paulis.components(matrix, dim=self._dim)
                data = qobj

            elif len(qobj.shape) == 1:
                # This is a 1D array of components
                try:
                    components = qobj.toarray()
                except AttributeError:
                    components = qobj

                qobj = components.reshape(np.square(self._dim))
                data = qobj

        else:
            self._dim = np.around(np.sqrt(qobj.shape)).astype(int)
            if not np.allclose(np.square(self._dim), qobj.shape):
                raise ValueError('qobj shape is invalid')

        return qobj, data

    def _add_labels(self, terms, mode):
        labels = paulis.labels(self._dim, symbol=self.symbol, delimiter=self.delimiter,
                               fmt=mode)

        # Update the term objects with the basis labels
        for term in terms:
            if mode == 'text':
                term.label = f'*{labels[term.index]}'
            else:
                term.label = str(labels[term.index])

    def _format_lhs(self, mode) -> Union[str, None]:
        return self.lhs_label
