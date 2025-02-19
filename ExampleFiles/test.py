import sympy
from sympy import simplify

x, y, a, b = sympy.symbols('x y a b')
f = a**3*x + 3*a**2*x**2/2 + a*x**3 + x**4/4
print(simplify(f.subs(a, 1)))
