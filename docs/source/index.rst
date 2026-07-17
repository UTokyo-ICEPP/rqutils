.. rqutils documentation master file, created by
   sphinx-quickstart on Wed Apr 27 09:06:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rqutils documentation
=====================

**rqutils** is a collection of Python modules that may be useful for quantum computing and
applications research.

Installation
------------

.. code-block:: console

      (.venv) $ pip install rqutils


.. toctree::
   :maxdepth: 1
   :caption: Available tools (API Reference):

   Pretty-printing quantum objects (state vectors and operators) <apidoc/rqutils.qprint>
   Scalable state vector simulator <apidoc/rqutils.svsim>
   Ground state solver (LOBPCG) <apidoc/rqutils.ground_locg>
   Sample-based quantum diagonalization <apidoc/rqutils.sqd>
   Miscellaneous math functions <apidoc/rqutils.math>
   Generalized Pauli (Gell-Mann) matrices <apidoc/rqutils.paulis.general>
   Symplectic representation of Pauli sums <apidoc/rqutils.paulis.symplectic>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
