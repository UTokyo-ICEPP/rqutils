from typing import Tuple, List, Union, Optional, Any
from numbers import Number
import builtins
from dataclasses import dataclass
from enum import Enum
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from qutip import Qobj
except ImportError:
    has_qutip = False
else:
    has_qutip = True
    
MATPLOTLIB_INLINE_BACKENDS = {
    "module://ipykernel.pylab.backend_inline",
    "module://matplotlib_inline.backend_inline",
    "nbAgg",
}

class QPrint:
    """Helper class to compose a LaTeX expression of a given quantum object.

    Args:
        qobj: Input quantum object.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        subsystem_dims: Specification of the dimensions of the subsystems.
        terms_per_row: Number of terms to show per row.
        binary: Show bra and ket indices in binary.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        epsilon: Numerical cutoff for ignoring amplitudes (relative to max) and phase (absolute).
        lhs_label: If not None, prepend '|`state_label`> = ' to the printout.
    """
    
    @dataclass
    class Term:
        sign: int
        amp: str
        phase: str
        ket: Optional[str] = None
        bra: Optional[str] = None
        
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
        subsystem_dims: Optional['array_like'] = None,
        terms_per_row: int = 0,
        binary: bool = False,
        amp_format: str = '.3f',
        phase_format: str = '.2f',
        epsilon: float = 1.e-6,
        lhs_label: Optional[str] = None
    ):
        self.qobj = qobj
        self.amp_norm = amp_norm
        self.phase_norm = phase_norm
        self.global_phase = global_phase
        self.subsystem_dims = subsystem_dims
        self.terms_per_row = terms_per_row
        self.binary = binary
        self.amp_format = amp_format
        self.phase_format = phase_format
        self.epsilon = epsilon
        self.lhs_label = lhs_label

    def compose(self) -> Tuple[int, str, str, List[List[Term]]]:
        """Compose a list of QPrintTerms."""
        
        qobj, data, objtype, dim = self._qobj_data()

        subsystem_dims = self.subsystem_dims
        if subsystem_dims is None:
            subsystem_dims = np.array([dim])
            
        assert np.prod(subsystem_dims) == dim, (f'Product of subsystem dimensions {np.prod(subsystem_dims)}'
                                                f' and qobj dimension {dim} do not match')

        # State label format template
        if self.binary:
            log2_dims = np.log2(np.asarray(subsystem_dims))
            assert np.allclose(log2_dims, np.round(log2_dims))
            label_template = ','.join(f'{{:0{s}b}}' for s in log2_dims.astype(int))
        else:
            label_template = ','.join(['{}'] * len(subsystem_dims))

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
        
        # Make tuples of quantum state labels and format the term indices
        if isinstance(qobj, scipy.sparse.csr_matrix):
            # CSR matrix: diff if indptr = number of elements in each row
            repeats = np.diff(qobj.indptr)
            row_labels_flat = np.repeat(np.arange(qobj.shape[0]), repeats)
            # unravel into row indices accounting for the tensor product
            if objtype in (QPrint.QobjType.KET, QPrint.QobjType.OPER):
                row_labels = np.unravel_index(row_labels_flat, subsystem_dims)
            if objtype in (QPrint.QobjType.BRA, QPrint.QobjType.OPER):
                col_labels = np.unravel_index(qobj.indices, subsystem_dims)
                
        elif isinstance(qobj, np.ndarray):
            if objtype in (QPrint.QobjType.KET, QPrint.QobjType.OPER):
                row_labels = np.unravel_index(np.arange(dim), subsystem_dims)
            if objtype in (QPrint.QobjType.BRA, QPrint.QobjType.OPER):
                col_labels = np.unravel_index(np.arange(dim), subsystem_dims)
            
        # List of terms
        terms = []

        for idx in term_indices:
            sign, phase_expr = sign_and_phase(norm_phase[idx], axis_proj[idx], rounded_phase[idx])
                
            if rounded_amp[idx] == -1:
                amp_expr = amp_template.format(absamp[idx])
            else:
                amp_expr = f'{rounded_amp[idx]}'
                
            term = QPrint.Term(sign=sign, amp=amp_expr, phase=phase_expr)

            if objtype in (QPrint.QobjType.KET, QPrint.QobjType.OPER):
                term.ket = label_template.format(*(r[idx[0]] for r in row_labels))
            if objtype in (QPrint.QobjType.BRA, QPrint.QobjType.OPER):
                # idx can be an 1- or 2-tuple depending on the type of self.qobj
                term.bra = label_template.format(*(c[idx[-1]] for c in col_labels))
                    
            terms.append(term)

        return global_sign, global_amp, global_phase, terms
    
    def _qobj_data(self):
        if has_qutip and isinstance(self.qobj, Qobj):
            qobj = self.qobj.data
            data = qobj.data
        elif isinstance(self.qobj, scipy.sparse.csr_matrix):
            qobj = self.qobj
            data = qobj.data
        elif isinstance(self.qobj, np.ndarray):
            qobj = self.qobj
            data = qobj
        else:
            raise NotImplementedError(f'qprint not implemented for {type(self.qobj)}')
        
        if len(qobj.shape) == 1 or qobj.shape[1] == 1:
            objtype = QPrint.QobjType.KET
            dim = qobj.shape[0]
        elif qobj.shape[0] == 1 and qobj.shape[1] != 1:
            objtype = QPrint.QobjType.BRA
            dim = qobj.shape[1]
        else:
            objtype = QPrint.QobjType.OPER
            dim = qobj.shape[0] # Only limiting to square matrices
            
        return qobj, data, objtype, dim
    
    def _format_phase(self, phase_expr):
        if phase_expr == '0':
            return ''
        elif phase_expr == '/':
            return 'i'
        
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
        
        return expr
    
    def __repr__(self):
        global_sign, global_amp, global_phase, terms = self.compose()
                
        expr = ''
        
        if global_sign == -1:
            expr += '-'
            
        expr += global_amp
        expr += self._format_phase(global_phase)

        if expr:
            has_pre_expr = True
            expr += ' ('
        else:
            has_pre_expr = False

        num_terms = 0
        
        for term in terms:
            num_terms += 1
            if num_terms == self.terms_per_row:
                if expr:
                    expr += '\n'
                num_terms = 0
            
            if expr:
                if term.sign == -1:
                    expr += ' - '
                else:
                    expr += ' + '
            elif term.sign == -1:
                expr += '-'
                
            if term.amp != '1':
                expr += term.amp
                
            expr += self._format_phase(term.phase)
            
            if term.ket is not None:
                expr += f'|{term.ket}>'
            if term.bra is not None:
                expr += f'<{term.bra}|'
                
        if has_pre_expr:
            expr += ')'
            
        if not expr:
            expr = '0'
            
        if self.lhs_label:
            _, _, objtype, _ = self._qobj_data()
            if objtype == QPrint.QobjType.KET:
                lhs = f'|{self.lhs_label}>'
            elif objtype == QPrint.QobjType.BRA:
                lhs = f'<{self.lhs_label}|'
            else:
                lhs = self.lhs_label
                
            return f'{lhs} = {expr}'
        else:
            return expr


class QPrintLaTeX(QPrint):
    def __init__(
        self,
        qobj: Any,
        amp_norm: Optional[Union[Number, Tuple[Number, str]]] = None,
        phase_norm: Tuple[Number, str] = (np.pi, '\pi'),
        global_phase: Optional[Union[Number, str]] = None,
        subsystem_dims: Optional['array_like'] = None,
        terms_per_row: int = 0,
        binary: bool = False,
        amp_format: str = '.3f',
        phase_format: str = '.2f',
        epsilon: float = 1.e-6,
        lhs_label: Optional[str] = None
    ):
        super().__init__(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            subsystem_dims=subsystem_dims,
            terms_per_row=terms_per_row,
            binary=binary,
            amp_format=amp_format,
            phase_format=phase_format,
            epsilon=epsilon,
            lhs_label=lhs_label)
    
    def _format_phase(self, phase_expr):
        if phase_expr == '0':
            return ''
        elif phase_expr == '/':
            return 'i'
        
        expr = 'e^{'
        
        if phase_expr != '1':
            expr += phase_expr
        
        if self.phase_norm is not None:
            if self.phase_norm[1] and self.phase_norm[1][0].isnumeric():
                expr += r' \cdot '
                    
            expr += self.phase_norm[1]

        expr += ' i}'
        
        return expr
    
    def _make_lines(self) -> list:
        global_sign, global_amp, global_phase, terms = self.compose()

        lines = []
        line_expr = ''
        num_terms = 0
        
        for term in terms:
            if term.sign == -1:
                line_expr += '-'
            elif lines or line_expr:
                line_expr += '+'
                
            if term.amp != '1':
                line_expr += term.amp
                
            line_expr += self._format_phase(term.phase)
            
            if term.ket is not None:
                line_expr += fr'| {term.ket} \rangle'
            if term.bra is not None:
                line_expr += fr'\langle {term.bra} |'
                
            num_terms += 1
            if num_terms == self.terms_per_row:
                lines.append(line_expr)
                line_expr = ''
                num_terms = 0
                
        if num_terms != 0:
            lines.append(line_expr)
            
        if not lines:
            lines = ['0']

        lhs = ''
        if self.lhs_label:
            _, _, objtype, _ = self._qobj_data()
            if objtype == QPrint.QobjType.KET:
                lhs = fr'| {self.lhs_label} \rangle'
            elif objtype == QPrint.QobjType.BRA:
                lhs = fr'\langle {self.lhs_label} |'
            else:
                lhs = self.lhs_label
                
        pre_expr = ''
        
        if global_sign == -1:
            pre_expr += '-'
        
        pre_expr += global_amp
        pre_expr += self._format_phase(global_phase)
            
        return lhs, pre_expr, lines
        
    def _repr_latex_(self) -> str:
        lhs, pre_expr, lines = self._make_lines()

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
                
        if lhs:
            lines[0] = f'{lhs} = {lines[0]}'
                
        if len(lines) == 1:
            return fr'$\displaystyle {lines[0]}$'
        else:
            return r'\begin{split} ' + r' \\ '.join(lines) + r' \end{split}'
            
    def mpl(self):
        lhs, pre_expr, lines = self._make_lines()
        
        if pre_expr:
            lines[0] = f'{pre_expr} ({lines[0]}'
            lines[-1] += ')'
            
        if lhs:
            lines[0] = f'{lhs} = {lines[0]}'
            
        fig, ax = plt.subplots(1, figsize=[10., 0.5 * len(lines)])
        ax.axis('off')
    
        num_rows = len(lines)
        for irow, line in enumerate(lines):
            ax.text(0.5, 1. / num_rows * (num_rows - irow - 1), f'${line}$', fontsize='x-large', ha='right')
            
        if mpl.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
            plt.close(fig)
                
        return fig


def qprint(qobj: Any, fmt: Optional[str] = None, **kwargs):
    if fmt is None:
        if hasattr(builtins, '__IPYTHON__'):
            fmt = 'latex'
        else:
            fmt = 'text'
            
    if fmt == 'text':
        return QPrint(qobj, **kwargs)
    elif fmt == 'latex':
        return QPrintLaTeX(qobj, **kwargs)
    elif fmt == 'mpl':
        return QPrintLaTeX(qobj, **kwargs).mpl()
    else:
        raise NotImplementedError(f'qprint with format {fmt} not implemented')
