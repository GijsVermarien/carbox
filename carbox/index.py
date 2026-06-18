import jax

@jax.tree_util.register_pytree_node_class
class Idx:
    def __init__(self, attrs):
        self.__dict__.update(attrs)

    def tree_flatten(self):
        keys = tuple(self.__dict__)
        values = tuple(self.__dict__.values())
        return values, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(dict(zip(keys, values)))