r"""A scalar-valued automatic differentiation engine.

This module implements backpropagation using reverse-mode automatic differentiation over a
dynamically built directed acyclic graph (DAG).

    Typical usage example:

    .. code:: python

        from pygrad.engine import Value

        a = Value(data=2.0)
        b = Value(data=3.0)
        c = Value(data=4.0)

        out = a + b
        out = out * c
        out = -out.tanh()

        out.backward()
"""
from __future__ import annotations

from math import tanh
from typing import Union


class Value:
    r"""This class represents a building block of a dynamically built directed acyclic graph.

    Value represents a node in a directed acyclic graph that is dynamically built during the
    execution of mathematical operations performed on an instance of Value.

    In this class `child_1` refers to node `self` and `child_2` to node `other`.
    Any operation accesses at least `child_1` attributes. For operations where only a single node
    is involved, `child_2` attributes remain `None`.

    Attributes:
        data: A float holding the node's value.
        grad: A float holding the node's gradient.
        child_1: Pointer from parent to first child node.
        child_2: Pointer from parent to second child node.
        grad_child_1: Gradient associated with first child.
        grad_child_2: Gradient associated with second child.
    """

    def __init__(self, data: float) -> None:
        """Initializes Value with provided data and zero gradient."""

        # Field for result of forward pass.
        self.data = data

        # Field for gradients computed during backward pass.
        self.grad = 0.0

        # Pointer to children of parent node.
        # Every parent node has exactly two children.
        self.child_1 = None
        self.child_2 = None

        # Store gradients of children in parent node.
        # These gradients depend on forward operation.
        # Necessary for backward pass.
        self.grad_child_1 = None
        self.grad_child_2 = None

    def backward(self) -> None:
        """Backward pass to compute gradients for each node of the computational graph.
        """
        # Set root node's gradient of directed acyclic graph to 1.0.
        self.grad = 1.0
        self.traverse(self.child_1, self.grad_child_1)
        self.traverse(self.child_2, self.grad_child_2)

    def traverse(self, other: Value, child_grad: float) -> None:
        """Traverses directed acyclic graph and computes gradients for each node.

        During the traverse of the computational graph all gradients from operations in which the
        child node was involved are summed up.

        Args:
            other: Child node.
            child_grad: Gradient associated with child.

        """
        if other is not None:
            other.grad += self.grad * child_grad
            other.traverse(other.child_1, other.grad_child_1)
            other.traverse(other.child_2, other.grad_child_2)

    def __add__(self, other: Value) -> Value:
        r"""Implements addition of nodes in a directed acyclic graph.

        .. math::
            f(x, y) = x + y \\
            \frac{df(x, y)}{dx} = 1 \\
            \frac{df(x, y)}{dy} = 1

        This operation adds two scalar values (forward pass), creates a parent node to store
        the result, adds the gradients associated with the operation to the newly created parent
        node, and adds two pointers from the parent node to its children.

        Args:
            other: Graph node.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = self.data + other.data
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradients for both children to parent node for addition operation.
        out.grad_child_1 = 1.0
        out.grad_child_2 = 1.0 
        # Add both children to parent node for tree traverse.
        out.child_1 = self
        out.child_2 = other
        return out

    def __sub__(self, other: Value) -> Value:
        r"""Implements subtraction of nodes in a directed acyclic graph.

        .. math::
            f(x, y) = x - y \\
            \frac{df(x, y)}{dx} = 1 \\
            \frac{df(x, y)}{dy} = -1

        This operation subtracts two scalar values (forward pass), creates a parent node to store
        the result, adds the gradients associated with the operation to the newly created parent
        node, and adds two pointers from the parent node to its children.

        Args:
            other: Graph node.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = self.data - other.data
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradients for both children to parent node for subtraction operation.
        out.grad_child_1 = 1.0
        out.grad_child_2 = -1.0
        # Add both children to parent node for tree traverse.
        out.child_1 = self
        out.child_2 = other
        return out

    def __mul__(self, other: Value) -> Value:
        r"""Implements multiplication of nodes in a directed acyclic graph.

        .. math::
            f(x, y) = x * y \\
            \frac{df(x, y)}{dx} = y \\
            \frac{df(x, y)}{dy} = x

        This operation multiplies two scalar values (forward pass), creates a parent node to store
        the result, adds the gradients associated with the operation to the newly created parent
        node, and adds two pointers from the parent node to its children.

        Args:
            other: Graph node.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = self.data * other.data
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradients for both children to parent node for multiplication operation.
        out.grad_child_1 = other.data
        out.grad_child_2 = self.data
        # Add both children to parent node for tree traverse.
        out.child_1 = self
        out.child_2 = other
        return out 

    def __truediv__(self, other: Value) -> Value:
        r"""Implements division of nodes in a directed acyclic graph.

        .. math::
            f(x, y) = x / y \\
            \frac{df(x, y)}{dx} = 1 / y \\
            \frac{df(x, y)}{dy} = - x / y^2

        This operation divides two scalar values (forward pass), creates a parent node to store
        the result, adds the gradients associated with the operation to the newly created parent
        node, and adds two pointers from the parent node to its children.

        Args:
            other: Graph node.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = self.data / other.data
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradients for both children to parent node for division operation.
        out.grad_child_1 = 1.0 / other.data
        out.grad_child_2 = -self.data / other.data**2
        # Add both children to parent node for tree traverse.
        out.child_1 = self
        out.child_2 = other
        return out

    def __pow__(self, power: Union[Value, float, int]) -> Value:
        r"""Implements power of a node in a directed acyclic graph.

        .. math::
            f(x; n) = x^n \\
            \frac{df(x)}{dx} = n * x^(n-1)

        This operation returns the power of a node (forward pass), creates a parent node to store
        the result, adds the gradient associated with the operation to the newly created parent
        node, and adds a pointer from the parent node to its child.

        Args:
            power: A float or an integer, the exponent.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = self.data ** power
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradient of child to parent node for power operation.
        out.grad_child_1 = power * self.data ** (power - 1)
        # Add child to parent node for tree traverse.
        out.child_1 = self
        return out

    def __neg__(self) -> Value:
        r"""Implements power of a node in a directed acyclic graph.

        .. math::
            f(x) = -1*x \\
            \frac{df(x)}{dx} = -1

        This operation returns the negation of a node (forward pass), creates a parent node to store
        the result, adds the gradient associated with the operation to the newly created parent
        node, and adds a pointer from the parent node to its child.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = -self.data
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradient of child to parent node for negation operation.
        out.grad_child_1 = -1.0
        # Add child to parent node for tree traverse.
        out.child_1 = self
        return out

    def tanh(self) -> Value:
        r"""Implements hyperbolic tangent of a node in a directed acyclic graph.

        .. math::
            f(x) = tanh(x) \\
            \frac{df(x)}{dx} = 1 - tanh(x)^2

        This operation returns the hyperbolic tangent of a node (forward pass), creates a parent
        node to store the result, adds the gradient associated with the operation to the newly
        created parent node, and adds a pointer from the parent node to its child.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = tanh(self.data)
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradient of child to parent node for negation tanh() operation.
        out.grad_child_1 = 1.0 - data**2
        # Add child to parent node for tree traverse.
        out.child_1 = self
        return out

    def relu(self) -> Value:
        r"""Implements ReLU of a node in a directed acyclic graph.

        .. math::
            f(x) = \text{ReLU}(x) =
            \begin{cases}
              x & x > 0\\
              0 & \text{else}\\
            \end{cases}

            \frac{df(x)}{dx} =
            \begin{cases}
              1 & x > 0\\
              0 & \text{else}\\
            \end{cases}


        This operation returns the ReLU of a node (forward pass), creates a parent node to store the
        result, adds the gradient associated with the operation to the newly created parent node,
        and adds a pointer from the parent node to its child.

        Returns:
            A new parent graph node.
        """
        # Forward pass.
        data = self.data * (self.data > 0)
        # Create new parent node in directed acyclic graph.
        out = Value(data=data)
        # Add gradient of child associated with operation to parent node.
        out.grad_child_1 = 1.0 * (self.data > 0)
        # Add child to parent node for tree traverse.
        out.child_1 = self
        return out

    def __repr__(self) -> str:
        return f"data = {self.data}\t grad = {self.grad}\t"
