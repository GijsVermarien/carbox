# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np 
import jax.numpy as jnp
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def load_network(fname):
    for row in open(fname):
        srow = row.strip()
        if srow == "":
            continue
        if srow[0] == "#":
            continue

        arow = srow.replace("[", "]").split("]")
        reaction, tlims, rate = [x.strip() for x in arow]

        if tlims.replace(" ", "") == "":
            tmin = 0e0
            tmax = 1e6
        else:
            tmin = float(tlims.split(",")[0])
            tmax = float(tlims.split(",")[1])

        reaction = reaction.replace("HE", "He")
        reaction = reaction.replace(" E", " e-")
        reaction = reaction.replace("E ", "e- ")
        reaction = reaction.replace("GRAIN0", "GRAIN")

        rr, pp = reaction.split("->")
        rr = [x.strip() for x in rr.split(" + ")]
        pp = [x.strip() for x in pp.split(" + ")]

        yield rr, pp, tmin, tmax, rate


class Reaction:
    def __init__(self, reactants, products, tmin, tmax, rate_str):
        self.reactants = reactants
        self.products = products
        self.tmin = tmin
        self.tmax = tmax
        self.rate_str = rate_str
        self.sympy_rate = parse_expr(rate_str, evaluate=True)
        self.rate_signature: list = None
        self.rate_mapping: dict = None

    def __repr__(self):
        return f"reaction({self.reactants}, {self.products}, {self.tmin}, {self.tmax}, {self.rate_str})"
reactions = [Reaction(*args) for args in load_network("../data/deuterated_clean.dat")]

reactions;


# +
def extract_operations(expr):
    """
    Extracts a list representing a sequential operation from a sympy expression,
    replacing numbers with dynamically generated variable names like float_a, float_b, etc.
    """
    operations = []
    float_counter = iter("abcdefghijklmnopqrstuvwxyz")  # Iterator for variable suffixes
    float_mapping = {}  # To store mapping of numbers to variable names

    # def traverse(e):
    for e in sp.preorder_traversal(expr):
        # print(e, e.func)
        if e.is_Atom:
            if e.is_number:  # Check if the atom is a number
                if e not in float_mapping:
                    # Assign a new variable name if not already mapped
                    float_mapping[e] = f"float_{next(float_counter)}"
                operations.append(float_mapping[e])
            else:
                operations.append(e)
        else:
            operations.append(e.func)
            # for arg in e.args:
            #     traverse(arg)

    # traverse(expr)
    return operations, float_mapping


# Extract operations and mappings for expr1 and expr2
operations1, float_mapping1 = extract_operations(reactions[998].sympy_rate)
operations2, float_mapping2 = extract_operations(reactions[997].sympy_rate)

print("Operations of expr1:", operations1)
print("Float mapping of expr1:", float_mapping1)
print("Operations of expr2:", operations2)
print("Float mapping of expr2:", float_mapping2)
# -

operations1 == operations2

with sp.evaluate(False):
    unique_reactions = []
    reaction_dict = {}
    for reaction in reactions:
        signature, float_mapping = extract_operations(reaction.sympy_rate)
        if tuple(signature) not in unique_reactions:
            unique_reactions.append(tuple(signature))
        reaction.rate_signature = signature
        reaction.rate_mapping = float_mapping


print(f"There are {len(unique_reactions)} unique reactions")


def extract_operations(expr):
    """
    Extracts a list representing a sequential operation from a sympy expression,
    replacing numbers with dynamically generated variable names like float_a, float_b, etc.
    """
    operations = []
    float_counter = iter("abcdefghijklmnopqrstuvwxyz")  # Iterator for variable suffixes
    float_mapping = {}  # To store mapping of numbers to variable names

    def traverse(e):
        print(e)
        if e.is_Atom:
            if e.is_number:  # Check if the atom is a number
                if e not in float_mapping:
                    # Assign a new variable name if not already mapped
                    float_mapping[e] = f"float_{next(float_counter)}"
                return float_mapping[e]
            else:
                return e
        else:
            print(e.func, e.args)
            return (e.func, tuple(traverse(arg) for arg in e.args))
                       
    operations = traverse(expr)
    float_mapping = {v: k for k, v in float_mapping.items()}
    return operations, float_mapping



def evaluate_expression(expr, mapping):
    """
    Evaluates a sympy expression using a mapping of variable names to values.
    """
    # print(expr, type(expr))
    if isinstance(expr, tuple):
        print("Path A", expr)
        func = expr[0]
        args = [evaluate_expression(e, mapping) for e in expr[1]]
        # args = [evaluate_expression(arg, mapping) for arg in expr[1]]
        print("Trying to evaluate: ", func, args)
        return func(*args)
    elif isinstance(expr, sp.Function):
        print("Path B", expr)
        print("Trying to evaluate: ", expr)
        return expr[0](*[evaluate_expression(arg, mapping) for arg in expr[1]])
    else:
        print("Path C", expr)
        return mapping.get(expr, expr)  # Return the value from mapping or the original expression


with sp.evaluate(False):
    tree_expr, float_mapping = extract_operations(reactions[998].sympy_rate)

# evaluate the expression tree directly:'
evaluate_expression(tree_expr, float_mapping)

# Check round trip works:
reactions[998].sympy_rate == evaluate_expression(tree_expr, float_mapping)




