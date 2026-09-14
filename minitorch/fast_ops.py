from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    # [
    #     [10, 20, 30],
    #     [40, 50, 60]
    # ]
    # in_storage = [10, 20, 30, 40, 50, 60]
    # in_shape = [2, 3]
    # in_strides = [3, 1]

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        same_shape = np.array_equal(out_shape, in_shape)
        same_strides = np.array_equal(out_strides, in_strides)

        # если совпадает и shape, и strides
        # значит элементы входа и результата расположены одинаково
        if same_shape and same_strides: 
            for ordinal in prange(len(out)):
                in_value = in_storage[ordinal]
                out[ordinal] = fn(in_value)
        # если формы или strides отличаются

        # f.e. 
        # input 
        #         [[10, 20, 30]]
        #output 
                # [
                #     [20, 40, 60],
                #     [20, 40, 60],
                # ]
        else:
            for i in prange(len(out)):
                out_index = np.empty(len(out_shape), dtype=np.int32)
                in_index = np.empty(len(in_shape), dtype=np.int32)

                # без этого возникает ошибка Overwrite of parallel loop index
                # и прикол в том, что Numba запрещает изменять переменную цикла prange, 
                # а  с помощью i + 0 создаём отдельное значение

                # и вот + 0 надо делать, 
                # так как j = i Numba считает просто вторым именем той же защищённой переменной цикла;
                # j = i + 0 создаёт новое вычисленное значение, 
                # которое уже можно изменять внутри to_index

                j = i + 0

                to_index(j, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_position = index_to_position(out_index, out_strides)
                in_position = index_to_position(in_index, in_strides)

                in_value = in_storage[in_position]
                out[out_position] = fn(in_value)

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # a:
        # [
        #     [1, 2, 3],
        #     [4, 5, 6],
        # ]
        #
        # b:
        # [[10, 20, 30]]
        #
        # out:
        # [
        #     [11, 22, 33],
        #     [14, 25, 36],
        # ]

        # одинаково ли расположены out и a
        same_a_shape = np.array_equal(out_shape, a_shape)
        same_a_strides = np.array_equal(out_strides, a_strides)

        # одинаково ли расположены out и b
        same_b_shape = np.array_equal(out_shape, b_shape)
        same_b_strides = np.array_equal(out_strides, b_strides)

        # все одинаково ли расположены
        all_same = (same_a_shape and same_a_strides and same_b_shape and same_b_strides)

        if all_same:
            for i in prange(len(out)):
                a_value = a_storage[i]
                b_value = b_storage[i]

                out[i] = fn(a_value, b_value)

        else:
            for i in prange(len(out)):
                out_index = np.empty(len(out_shape), dtype=np.int32)
                a_index = np.empty(len(a_shape), dtype=np.int32)
                b_index = np.empty(len(b_shape), dtype=np.int32)

                j = i + 0

                to_index(j, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                out_position = index_to_position(out_index, out_strides)
                a_position = index_to_position(a_index, a_strides)
                b_position = index_to_position(b_index, b_strides)

                a_value = a_storage[a_position]
                b_value = b_storage[b_position]

                out[out_position] = fn(a_value, b_value)

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        #
        # a:
        # [
        #     [1, 2, 3],
        #     [4, 5, 6],
        # ]
        #
        # reduce_dim = 1
        #
        # out:
        # [
        #     [6],
        #     [15],
        # ]

        for i in prange(len(out)):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            a_index = np.empty(len(a_shape), dtype=np.int32)

            j = i + 0

            # индекс элемента результата
            to_index(j, out_shape, out_index)

            # копируем индекс результата во входной индекс
            for dim in range(len(a_shape)):
                a_index[dim] = out_index[dim]

            a_index[reduce_dim] = 0

            out_position = index_to_position(out_index, out_strides)

            # получаем начальную позицию во входном тензоре
            a_position = index_to_position(a_index, a_strides)

            result = out[out_position]

            # сколько элементов нужно объединить
            reduce_size = a_shape[reduce_dim]

            # и на сколько позиций двигаться по storage
            reduce_stride = a_strides[reduce_dim]

            for reduce_i in range(reduce_size):
                a_value = a_storage[a_position]
                result = fn(result, a_value)
                a_position = a_position + reduce_stride

            out[out_position] = result

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    out_rows = out_shape[1]
    out_columns = out_shape[2]

    # колво умножений для одного элемента out
    inner_size = a_shape[2]

    for i in prange(len(out)):
        batch = i // (out_rows * out_columns)
        row = (i // out_columns) % out_rows
        column = i % out_columns

        out_position = batch * out_strides[0] + row * out_strides[1] + column * out_strides[2]
        a_position = batch * a_batch_stride + row * a_strides[1]
        b_position = batch * b_batch_stride + column * b_strides[2]

        result = 0.0

        for _ in range(inner_size):
            result = result + a_storage[a_position] * b_storage[b_position]
            a_position = a_position + a_strides[2]
            b_position = b_position + b_strides[1]

        out[out_position] = result


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
