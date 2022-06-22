import jax
import torch

from pygrad.engine import Value

places = 5


def test_1_autograd():

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
    f = Value(data=_f)

    out = fun(a, b, c, d, e, f)
    out.backward()

    out_pg, a_pg, b_pg, c_pg, d_pg, e_pg, f_pg = out, a, b, c, d, e, f

    # PyTorch
    a = torch.tensor([_a], requires_grad=True)
    b = torch.tensor([_b], requires_grad=True)
    c = torch.tensor([_c], requires_grad=True)
    d = torch.tensor([_d], requires_grad=True)
    e = torch.tensor([_e], requires_grad=True)
    f = torch.tensor([_f], requires_grad=True)

    out = fun(a, b, c, d, e, f)
    out.backward()

    out_pt, a_pt, b_pt, c_pt, d_pt, e_pt, f_pt = out, a, b, c, d, e, f

    # Assert correct forward pass
    assert round(out_pt.data.item() - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_pt.grad.item() - a_pg.grad, places) == 0
    assert round(b_pt.grad.item() - b_pg.grad, places) == 0
    assert round(c_pt.grad.item() - c_pg.grad, places) == 0
    assert round(d_pt.grad.item() - d_pg.grad, places) == 0
    assert round(e_pt.grad.item() - e_pg.grad, places) == 0
    assert round(f_pt.grad.item() - f_pg.grad, places) == 0


def test_2_autograd():

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
    f = Value(data=_f)

    out = fun(a, b, c, d, e, f)
    out.backward()

    out_pg, a_pg, b_pg, c_pg, d_pg, e_pg, f_pg = out, a, b, c, d, e, f

    # Jax
    a = _a
    b = _b
    c = _c
    d = _d
    e = _e
    f = _f

    out = fun(a, b, c, d, e, f)

    a_grad, b_grad, c_grad, d_grad, e_grad, f_grad = \
        jax.grad(fun, argnums=(0, 1, 2, 3, 4, 5))(a, b, c, d, e, f)

    # Assert correct forward pass
    assert round(out - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0
    assert round(c_grad - c_pg.grad, places) == 0
    assert round(d_grad - d_pg.grad, places) == 0
    assert round(e_grad - e_pg.grad, places) == 0
    assert round(f_grad - f_pg.grad, places) == 0


def test_3_autograd():

    def fun(a_, b_, c_, d_, e_, f_):
        out_ = (a_ + b_).tanh()
        out_ = out_ - c_
        out_ = out_ * d_
        out_ = out_ / e_
        out_ = out_ ** f_
        return out_

    _a = 2.0
    _b = 3.0
    _c = 4.0

    # PyGrad
    a = Value(data=_a)
    b = Value(data=_b)
    c = _c

    out = fun(a, a, b, a, b, c)
    out.backward()

    out_pg, a_pg, b_pg, c_pg = out, a, b, c

    # PyTorch
    a = torch.tensor([_a], requires_grad=True)
    b = torch.tensor([_b], requires_grad=True)
    c = _c

    out = fun(a, a, b, a, b, c)
    out.backward()

    out_pt, a_pt, b_pt, c_pt = out, a, b, c

    # Assert correct forward pass
    assert round(out_pt.data.item() - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_pt.grad.item() - a_pg.grad, places) == 0
    assert round(b_pt.grad.item() - b_pg.grad, places) == 0


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

    f = lambda x, y: x + y
    out_data = f(a, b)
    a_grad, b_grad = jax.grad(f, argnums=(0, 1))(a, b)

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

    f = lambda x, y: x - y
    out_data = f(a, b)
    a_grad, b_grad = jax.grad(f, argnums=(0, 1))(a, b)

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

    f = lambda x, y: x * y
    out_data = f(a, b)
    a_grad, b_grad = jax.grad(f, argnums=(0, 1))(a, b)

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

    f = lambda x, y: x / y
    out_data = f(a, b)
    a_grad, b_grad = jax.grad(f, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_pow_1():

    a_ = 2.0
    b_ = 3.0

    # PyGrad
    a = Value(data=a_)
    b = b_

    out = a ** b
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    f = lambda x, y: x ** y
    out_data = f(a, b)
    a_grad, b_grad = jax.grad(f, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    # assert round(b_grad - b_pg.grad, places) == 0


def test_pow_2():

    a_ = 2.0
    b_ = 3.0

    # PyGrad
    a = Value(data=a_)
    b = Value(data=b_)

    out = a ** b
    out.backward()

    out_pg, a_pg, b_pg = out, a, b

    # Jax
    a = a_
    b = b_

    f = lambda x, y: x ** y
    out_data = f(a, b)
    a_grad, b_grad = jax.grad(f, argnums=(0, 1))(a, b)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
    assert round(b_grad - b_pg.grad, places) == 0


def test_tanh():

    a_ = 2.0

    # PyGrad
    a = Value(data=a_)

    out = a.tanh()
    out.backward()

    out_pg, a_pg= out, a

    # Jax
    a = a_

    f = lambda x: jax.numpy.tanh(x)
    out_data = f(a)
    a_grad = jax.grad(f, argnums=0)(a)

    # Assert correct forward pass
    assert round(out_data - out_pg.data, places) == 0

    # Assert correct gradients
    assert round(a_grad - a_pg.grad, places) == 0
