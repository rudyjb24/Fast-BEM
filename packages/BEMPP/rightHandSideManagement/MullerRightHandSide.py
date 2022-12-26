from packages.physicalObjects.physicalValues import *


def getMullerRightHandSide(bubbles, dirichletFunction, neumannFunction):
    proyections = []

    for bubble in bubbles:
        dirichletGridFunction = bempp.api.GridFunction(
            bubble.functionSpace, fun=dirichletFunction
        )
        neumannGridFunction = bempp.api.GridFunction(
            bubble.functionSpace, fun=neumannFunction
        )
        proyections.append(dirichletGridFunction.projections())
        proyections.append(neumannGridFunction.projections())

    rightHandSide = np.concatenate([proyection for proyection in proyections])

    return rightHandSide
