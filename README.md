# pygrad

**Work in progress.**

A minimal*, fast**, scalar-valued, dependency-free Python library for automatic differentiation***.

## Automatic differentiation

*pygrad* implements backpropagation using 
[reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation) over a dynamically
built directed acyclic graph. The `Value` class represents a scalar-valued node in a directed 
acyclic graph holding the information about its value, the associated gradient to the value as well 
as the type of operation from which it was created. 

A new parent node is created for each mathematical operation and added to the computational graph. 
Each operation includes at least one but not more than two child nodes. Each operation populates the 
`Value`'s `data` field as well as references to the child nodes required for the backward pass.

## Installation

To use *pygrad*, first install it using *pip*:

```commandline
cd PyAutograd
pip install . 
```

## Example Usage

```python
from pygrad import Value

a = Value(data=2.0)
b = Value(data=3.0)
c = Value(data=4.0)
d = Value(data=5.0)
e = Value(data=6.0)
f = Value(data=2.0)

out = a + b
out = out - c
out = out * d
out = out / e
out = out ** f
out = out.tanh()

out.backward()

print(out)  # prints out.data = 0.6008296026925349, out.grad = 1.0 
print(a)    # prints a.data = 2.0, a.grad = 0.8875052618449036
print(b)    # prints b.data = 3.0, b.grad = 0.8875052618449036
print(c)    # prints c.data = 4.0, c.grad = -0.8875052618449036 
# ...
```

## Documentation

Run the following command to generate the documentation:

```commandline
cd docs
sphinx-apidoc -f -o source/ ../pygrad && make html
```

## Tests

Yes, there are some tests. But they are by no means complete. The tests use JAX's autograd engine as
a reference. You can run them by typing

```commandline
cd PyAutograd 
pytest tests
```

## TODO

- Add private methods to Sphinx documentation.
- Add image to visualize working principle of backpropagation using DAGs.


## Licence

MIT

---
*It's really bare-bones. Just have a look at *engine.py*.

**In Python terms.

***Use with caution.
