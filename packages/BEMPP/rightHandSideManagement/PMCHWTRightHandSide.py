from packages.physicalObjects.physicalValues import *


def getPMCHWTRightHandSide(bubbles, dirichletFunction, neumannFunction):
    stuff = []

    for bubble in bubbles:
        # revisar si bempp paraleliza esto
        dirichletGridFunction = bempp.api.GridFunction(
            bubble.functionSpace, fun=dirichletFunction
        )
        neumannGridFunction = bempp.api.GridFunction(
            bubble.functionSpace, fun=neumannFunction
        )
        stuff.append(dirichletGridFunction.coefficients) # probablemente usando weak from -> cambiarlo por strong form
        stuff.append(neumannGridFunction.coefficients) #projections -> coefficients
    #implementar precondicionador: (Matriz Masa)^(-1)
    #functionSpace se le puede pedir masa
    rightHandSide = np.concatenate([coefficient for coefficient in stuff])

    return rightHandSide
