from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    delta_x_minus = []
    delta_x_plus = []

    for index, value in enumerate(vals):
        if index == arg:
            delta_x_minus.append(value - epsilon)
            delta_x_plus.append(value + epsilon)
        else:
            delta_x_minus.append(value)
            delta_x_plus.append(value)

    return (f(*delta_x_plus) - f(*delta_x_minus)) / (2 * epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


# тут юзала https://algorithmica.org/ru/dfs
def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order = []
    used = set()

    def dfs(current: Variable) -> None:
        if current.is_constant():
            return

        if current.unique_id in used:
            return

        used.add(current.unique_id)

        for parent in current.parents:
            dfs(parent)

        order.append(current)

    dfs(variable)
    order.reverse()

    return order



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # все найденные переменные
    ds = {variable.unique_id: deriv}

    # порядок обхода <-
    order = topological_sort(variable)

    for v in order: # идем по всем позициям в графе
        num = v.unique_id # берем ее номер (типо 1 - самая правая и тд)
        u = ds[num] # берем ее производную

        # если мы пришли в конец (самая левая позиция)
        if v.is_leaf():
            v.accumulate_derivative(u) # взяли накоп произв
            continue

        # найдем произв объектов, которые левее и от которых зависим
        left_point = v.chain_rule(u)

        for x, x_der in left_point:
            x_num = x.unique_id

            # проделываем тоже самое с левыми объектами
            if x_num not in ds:
                ds[x_num] = 0.0

            ds[x_num] += x_der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
