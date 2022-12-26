from packages.physicalObjects.physicalValues import *
from .NeumannRightHandSide import getNeumannRightHandSide
from .DirichletRightHandSide import getDirichletRightHandSide
from .PMCHWTRightHandSide import getPMCHWTRightHandSide
from .MullerRightHandSide import getMullerRightHandSide

wavenumber = None


def getRightHandSide(formulationType, bubbles, logs):
    t0 = tm.time()
    global wavenumber
    wavenumber = bubbles[0].exteriorWavenumber

    @bempp.api.complex_callable
    def dirichletFunction(x, n, domain_index, result):
        result[0] = np.exp(1j * wavenumber * x[0])

    @bempp.api.complex_callable
    def neumannFunction(x, n, domain_index, result):
        result[0] = 1j * wavenumber * n[0] * np.exp(1j * wavenumber * x[0])

    if formulationType == 0:
        rightHandSide = getDirichletRightHandSide(bubbles, dirichletFunction)
    elif formulationType == 1:
        rightHandSide = getNeumannRightHandSide(bubbles, neumannFunction)
    elif formulationType == 2:
        rightHandSide = getPMCHWTRightHandSide(
            bubbles, dirichletFunction, neumannFunction
        )
    elif formulationType == 3:
        rightHandSide = getMullerRightHandSide(
            bubbles, dirichletFunction, neumannFunction
        )

    t1 = tm.time()
    if logs:
        print("Right hand side:{}{}s".format(" " * 7, round(t1 - t0, 3)))
    return rightHandSide
