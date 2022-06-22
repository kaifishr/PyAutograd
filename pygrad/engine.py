from __future__ import annotations

from math import log, tanh
from typing import Union


class Value:

    # Global counter
    _id = 0

    def __init__(self, data: float) -> None:

        self.data = data
        self.grad = 0.0

        # Connects parent node with its children.
        # Every parent node has exactly two children.
        self.child_1 = None
        self.child_2 = None

        # Store gradients of children in parent node.
        # Necessary for backward pass.
        # These gradients depend on forward operation.
        self.grad_child_1 = None
        self.grad_child_2 = None

        # Later used to create dot graph.
        self._operation = None

        # Assign unique ID to object
        self._id = Value._id
        Value._id += 1 

    def backward(self) -> None:
        print("\nBackward")
        # Set initial gradient to 1.0
        self.grad = 1.0
        # Perform width-first tree traverse
        # print(self)
        self.traverse(self.child_1, self.grad_child_1)
        self.traverse(self.child_2, self.grad_child_2)

    def traverse(self, other: Value, child_grad: float) -> None:
        if other is not None:
            # print(other)
            other.grad += self.grad * child_grad  # gradient of parent * gradient of child
            other.traverse(other.child_1, other.grad_child_1)
            other.traverse(other.child_2, other.grad_child_2)

    def __add__(self, other: Value) -> Value:
        print("Add")
        # Forward pass
        data = self.data + other.data
        # Create new parent node
        out = Value(data=data)
        # Add gradients for both children to parent node for addition operation.
        # This is required to compute the gradients during the backward pass.
        out.grad_child_1 = 1.0
        out.grad_child_2 = 1.0 
        # Add both children to parent node for tree traverse
        out.child_1 = self
        out.child_2 = other
        return out

    def __sub__(self, other: Value) -> Value:
        print("Sub")
        # Forward pass
        data = self.data - other.data
        # Create new parent node
        out = Value(data=data)
        # Add gradients for both children to parent node for addition operation.
        # This is required to compute the gradients during the backward pass.
        out.grad_child_1 = 1.0
        out.grad_child_2 = -1.0
        # Add both children to parent node for tree traverse
        out.child_1 = self
        out.child_2 = other
        return out

    def __mul__(self, other: Value) -> Value:
        print("Mul")
        # Forward pass
        data = self.data * other.data
        # Create new parent node
        out = Value(data=data)
        # Add gradients for both children to parent node for multiplication operation.
        # This is required to compute the gradients during the backward pass.
        out.grad_child_1 = other.data
        out.grad_child_2 = self.data
        # Add both children to parent node for tree traverse
        out.child_1 = self
        out.child_2 = other
        return out 

    def __truediv__(self, other: Value) -> Value:
        print("Div")
        # Forward pass
        data = self.data / other.data
        # Create new parent node
        out = Value(data=data)
        # Add gradients for both children to parent node for multiplication operation.
        # This is required to compute the gradients during the backward pass.
        # 'child_1' represents 'self'
        # 'child_2' represents 'other'
        out.grad_child_1 = 1.0 / other.data
        out.grad_child_2 = -self.data / other.data**2
        # Add both children to parent node for tree traverse
        out.child_1 = self
        out.child_2 = other
        return out

    def __pow__(self, power: Union[Value, float, int], modulo=None):
        """

        Args:
            power:
            modulo:

        Returns:

        """
        print("Pow")
        if isinstance(power, Value):
            # Forward pass
            data = self.data ** power.data
            # Create new parent node in directed acyclic graph (DAG)
            out = Value(data=data)
            # Add gradients for both children to parent node for power operation.
            # This is required to compute the gradient during the backward pass.
            # 'child_1' belongs to 'self'
            # 'child_2' belongs to 'power'
            out.grad_child_1 = power.data * self.data ** (power.data - 1)
            out.grad_child_2 = data * log(self.data)  # note: this can crash --> remove
            # Add both children to parent node for tree traverse
            out.child_1 = self
            out.child_2 = power
            return out
        else:
            # Forward pass
            data = self.data ** power
            # Create new parent node in directed acyclic graph (DAG)
            out = Value(data=data)
            # Add gradients of child to parent node for power operation.
            # This is required to compute the gradients during the backward pass.
            # 'child_1' belongs to 'self'
            out.grad_child_1 = power * self.data ** (power - 1)
            out.grad_child_2 = None  # just for completeness
            # Add child to parent node for tree traverse
            out.child_1 = self
            return out

    def tanh(self) -> Value:
        """Hyperbolic tangent.

        Returns:
            A new value with the hyperbolic tangent of self.data.
        """
        # Forward pass
        data = tanh(self.data)
        # Create new parent node in directed acyclic graph (DAG)
        out = Value(data=data)
        # Add gradients of child (self) to parent node (out) for tanh()-operation.
        # This is required to compute the gradient during the backward pass.
        # 'child_1' belongs to 'self'
        out.grad_child_1 = 1.0 - data**2
        out.grad_child_2 = None
        # Add child to parent node (out) for tree traverse
        out.child_1 = self
        return out

    def __repr__(self):
        return f"data = {self.data}\t grad = {self.grad}\t " \
               f"grad_child_1 = {self.grad_child_1}\t " \
               f"grad_child_2 = {self.grad_child_2}\t " \
               f"id(child_1) = {self.child_1._id if self.child_1 else None}\t " \
               f"id(child_2) = {self.child_2._id if self.child_2 else None}\t " \
               f"id = {self._id}"

