tensorsat.diagrams
==================

.. automodule:: tensorsat.diagrams

Block
-----

Type alias for a block in a diagram, which can be either:

- a box, as an instance of a subclass of :class:`Box`;
- a sub-diagram, as an instance of :class:`Diagram`.

.. autodata:: tensorsat.diagrams.Block

Box
---

.. autoclass:: tensorsat.diagrams.Box
    :show-inheritance:
    :members:
    :special-members: __new__

BoxClass
--------

.. autodata:: tensorsat.diagrams.BoxClass

BoxMeta
-------

.. autoclass:: tensorsat.diagrams.BoxMeta
    :show-inheritance:
    :members:

BoxT_inv
--------

.. autodata:: tensorsat.diagrams.BoxT_inv

Diagram
-------

.. autoclass:: tensorsat.diagrams.Diagram
    :show-inheritance:
    :members:
    :special-members: __rshift__, __rrshift__, __new__

DiagramBuilder
--------------

.. autoclass:: tensorsat.diagrams.DiagramBuilder
    :show-inheritance:
    :members:
    :special-members: __rmatmul__, __new__, __getitem__

DiagramRecipe
-------------

.. autoclass:: tensorsat.diagrams.DiagramRecipe
    :show-inheritance:
    :members:
    :special-members: __call__

Port
----

.. autoclass:: tensorsat.diagrams.Port
    :show-inheritance:
    :members:

PortOrderStructure
------------------

.. autoclass:: tensorsat.diagrams.PortOrderStructure
    :show-inheritance:
    :members:
    :special-members: __new__

Ports
-----

.. autodata:: tensorsat.diagrams.Ports

RecipeParams
------------

.. autodata:: tensorsat.diagrams.RecipeParams

SelectedBlockPorts
------------------

.. autoclass:: tensorsat.diagrams.SelectedBlockPorts
    :show-inheritance:
    :members:
    :special-members: __rshift__

SelectedBuilderWires
--------------------

.. autoclass:: tensorsat.diagrams.SelectedBuilderWires
    :show-inheritance:
    :members:
    :special-members: __rmatmul__

Shape
-----

.. autodata:: tensorsat.diagrams.Shape

Shaped
------

.. autoclass:: tensorsat.diagrams.Shaped
    :show-inheritance:
    :members:

Slot
----

.. autoclass:: tensorsat.diagrams.Slot
    :show-inheritance:
    :members:

Slots
-----

.. autodata:: tensorsat.diagrams.Slots

TYPE_CHECKING
-------------

.. autodata:: tensorsat.diagrams.TYPE_CHECKING

TensorLikeBox
-------------

.. autoclass:: tensorsat.diagrams.TensorLikeBox
    :show-inheritance:
    :members:
    :special-members: __new__

TensorLikeBoxT_inv
------------------

.. autodata:: tensorsat.diagrams.TensorLikeBoxT_inv

TensorLikeType
--------------

.. autoclass:: tensorsat.diagrams.TensorLikeType
    :show-inheritance:
    :members:
    :special-members: __new__

Type
----

.. autoclass:: tensorsat.diagrams.Type
    :show-inheritance:
    :members:
    :special-members: __new__

TypeMeta
--------

.. autoclass:: tensorsat.diagrams.TypeMeta
    :show-inheritance:
    :members:

Wire
----

.. autoclass:: tensorsat.diagrams.Wire
    :show-inheritance:
    :members:

Wires
-----

.. autodata:: tensorsat.diagrams.Wires

Wiring
------

.. autoclass:: tensorsat.diagrams.Wiring
    :show-inheritance:
    :members:
    :special-members: __new__

WiringBuilder
-------------

.. autoclass:: tensorsat.diagrams.WiringBuilder
    :show-inheritance:
    :members:
    :special-members: __new__

WiringData
----------

.. autoclass:: tensorsat.diagrams.WiringData
    :show-inheritance:
    :members:
