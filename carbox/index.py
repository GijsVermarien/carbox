import jax


@jax.tree_util.register_pytree_node_class
class Idx:
    """
    Static species-name -> array-index lookup (e.g. ``idx.H`` gives the
    position of "H" in the abundance vector).

    Registered as a pytree with zero leaves: the name->index mapping is
    carried entirely as aux_data, so it adds no traced values and is free
    under jit/vmap (each ``idx.<name>`` access is a plain Python int, i.e. a
    static index).
    """

    def __init__(self, name_to_index):
        self._map = dict(name_to_index)

    def __getattr__(self, name):
        try:
            return self._map[name]
        except KeyError:
            raise AttributeError(name)

    def tree_flatten(self):
        return (), tuple(sorted(self._map.items()))

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(dict(aux_data))
