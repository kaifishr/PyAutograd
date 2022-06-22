# pygrad

A minimal*, fast**, scalar-valued, dependency-free Python library for automatic differentiation***.

*pygrad* implements backpropagation using reverse-mode automatic differentiation over a dynamically
built directed acyclic graph.

## Example Usage

```python
from pygrad.engine import Value

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



## Tests

Yes, there are some tests. But they are by no means complete. You can run them by typing

```commandline
cd autograd
pytest tests
```

## Licence

MIT

---
*It's really bare-bones. Just have a look in *engine.py*.

**Relatively speaking. In Python terms.

***Use with caution.