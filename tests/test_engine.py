import jax
import torch

from pygrad.engine import Value

places = 5


def test_autograd_1():

    def fun(a_, b_, c_, d_, e_, f_):
        out_ = a_ + b_
        out_ = out_ - c_
        out_ = out_ * d_
        out_ = out_ / e_
        out_ = out_ ** f_
        return out_

    _a = 2.0
    _b = 3.0
    _c = 4.0
    _d = 5.0
    _e = 6.0
    _f = 2.0

    # PyGrad
    a = Value(data=_a)
    b = Value(data=_b)
    c = Value(data=_c)
    d = Value(data=_d)
    e = Value(data=_e)
    f = _f

    out = fun(a, b, c, d, e, f)
    out.backward()

    out_pg, a_pg, b_pg, c_pg, d_pg, e_pg = out, a, b, c, d, e

    # Jax
    a = _a
    b = _b
    c = _c
    d = _d
    e = _e
    f = _f

    out = fun(a, b, c, d, e, f)

    a_grad, b_grad, c_grad, d_grad, e_grad = jax.grad(fun, argnums=(0, 1, 2, 3, 4))(a, b, c, d, e, f)

    # Assert correct forward pass
    assert round(out - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0
    assert round(c_grad - c_pg.grad, places) == 0
    assert round(d_grad - d_pg.grad, places) == 0
    assert round(e_grad - e_pg.grad, places) == 0


def test_autograd_2():

    def fun_pg(a_, b_, c_):
        out_ = (a_ + b_).tanh()
        out_ = out_ - a_
        out_ = out_ * b_
        out_ = out_ / a_
        out_ = out_ ** c_
        return out_

    def fun_jx(a_, b_, c_):
        out_ = jax.numpy.tanh(a_ + b_)
        out_ = out_ - a_
        out_ = out_ * b_
        out_ = out_ / a_
        out_ = out_ ** c_
        return out_

    _a = 2.0
    _b = 3.0
    _c = 4.0

    # PyGrad
    a = Value(data=_a)
    b = Value(data=_b)
    c = _c

    out = fun_pg(a, b, c)
    out.backward()

    out_pg, a_pg, b_pg, c_pg = out, a, b, c

    # Jax
    a = _a
    b = _b
    c = _c

    out = fun_jx(a, b, c)

    a_grad, b_grad, c_grad = jax.grad(fun_jx, argnums=(0, 1, 2))(a, b, c)

    # Assert correct forward pass
    assert round(out - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_autograd_3():

    def fun(a_, b_):
        c = a_ - b_
        d = a_ * b_
        e = c / d
        f = e - d
        return f

    _a = -3.0
    _b = 5.0

    # PyGrad
    a = Value(data=_a)
    b = Value(data=_b)

    out = fun(a, b)
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = _a
    b = _b

    out = fun(a, b)

    a_grad, b_grad = jax.grad(fun, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_autograd_4():

    def fun(x, y):
        z = y * x + y + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        return y

    _x = -4.0
    _y = 2.0

    # PyGrad
    x_ = Value(_x)
    y_ = Value(_y)

    out = fun(x_, y_)
    out.backward()

    out_pg, x_pg, y_pg = out, x_, y_

    # PyTorch
    x_ = torch.tensor([_x]).double()
    y_ = torch.tensor([_y]).double()
    x_.requires_grad = True
    y_.requires_grad = True

    out = fun(x_, y_)
    out.backward()

    out_pt, x_pt, y_pt = out, x_, y_

    # Assert correct forward pass
    assert round(out.data.item() - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(x_pt.grad.data.item() - x_pg.grad, places) == 0
    assert round(y_pt.grad.data.item() - y_pg.grad, places) == 0


def test_autograd_5():

    def fun(a, b, u, v, w, x):
        c = a + b
        d = a * b + b ** 3
        c = c + c + u
        c = c + u + c + (-a)
        d = d + d * v + (b + a).relu()
        d = d + w * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / v
        g = g + x / f
        return g

    _a = -3.0
    _b = 2.0
    _u = 1.0
    _v = 2.0
    _w = 3.0
    _x = 10.0

    # PyGrad
    a_ = Value(_a)
    b_ = Value(_b)
    u_ = Value(_u)
    v_ = Value(_v)
    w_ = Value(_w)
    x_ = Value(_x)

    out = fun(a_, b_, u_, v_, w_, x_)
    out.backward()

    out_pg, a_pg, b_pg = out, a_, b_

    # PyTorch
    a_ = torch.tensor([_a]).double()
    b_ = torch.tensor([_b]).double()
    u_ = torch.tensor([_u]).double()
    v_ = torch.tensor([_v]).double()
    w_ = torch.tensor([_w]).double()
    x_ = torch.tensor([_x]).double()
    a_.requires_grad = True
    b_.requires_grad = True

    out = fun(a_, b_, u_, v_, w_, x_)
    out.backward()

    out_pt, a_pt, b_pt = out, a_, b_

    # Assert correct forward pass
    assert round(out.data.item() - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_pt.grad.data.item() - a_pg.grad, places) == 0
    assert round(b_pt.grad.data.item() - b_pg.grad, places) == 0


def test_add():

    a_ = 2.0
    b_ = -3.0

    # PyGrad
    a = Value(data=a_)
    b = Value(data=b_)

    out = a + b
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    fun = lambda x, y: x + y
    out_data = fun(a, b)
    a_grad, b_grad = jax.grad(fun, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_sub():

    a_ = 2.0
    b_ = -3.0

    # PyGrad
    a = Value(data=a_)
    b = Value(data=b_)

    out = a - b
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    fun = lambda x, y: x - y
    out_data = fun(a, b)
    a_grad, b_grad = jax.grad(fun, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_mul():

    a_ = 2.0
    b_ = -3.0

    # PyGrad
    a = Value(data=a_)
    b = Value(data=b_)

    out = a * b
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    fun = lambda x, y: x * y
    out_data = fun(a, b)
    a_grad, b_grad = jax.grad(fun, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_div():

    a_ = 2.0
    b_ = -3.0

    # PyGrad
    a = Value(data=a_)
    b = Value(data=b_)

    out = a / b
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    fun = lambda x, y: x / y
    out_data = fun(a, b)
    a_grad, b_grad = jax.grad(fun, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_pow():

    a_ = 2.0
    b_ = 3.0

    # PyGrad
    a = Value(data=a_)
    b = b_

    out = a ** b  # pow(a, b)
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    fun = lambda x, y: x ** y
    out_data = fun(a, b)
    a_grad, b_grad = jax.grad(fun, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    # assert round(b_grad - b_pg.grad, places) == 0


def test_neg():
    """Tests __neg__() method of Value.
    """
    a_ = 2.0

    # PyGrad
    a = Value(data=a_)

    out = -a
    out.backward()

    out_pg, a_pg= out, a

    # Jax
    a = a_

    fun = lambda x: -x
    out_data = fun(a)
    a_grad = jax.grad(fun, argnums=0)(a)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0


def test_tanh():

    a_ = 2.0

    # PyGrad
    a = Value(data=a_)

    out = a.tanh()
    out.backward()

    out_pg, a_pg = out, a

    # Jax
    a = a_

    fun = lambda x: jax.numpy.tanh(x)
    out_data = fun(a)
    a_grad = jax.grad(fun, argnums=0)(a)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0


def test_relu_1():

    a_ = 2.0

    # PyGrad
    a = Value(data=a_)

    out = a.relu()
    out.backward()

    out_pg, a_pg = out, a

    # Jax
    a = a_

    fun = lambda x: x * (x > 0)
    out_data = fun(a)
    a_grad = jax.grad(fun, argnums=0)(a)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0


def test_relu_2():

    a_ = -2.0

    # PyGrad
    a = Value(data=a_)

    out = a.relu()
    out.backward()

    out_pg, a_pg = out, a

    # Jax
    a = a_

    fun = lambda x: x * (x > 0)
    out_data = fun(a)
    a_grad = jax.grad(fun, argnums=0)(a)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
