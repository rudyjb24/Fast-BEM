from packages.physicalObjects.physicalValues import *


def getDirichletRightHandSide(bubbles, dirichletFunction):
    proyections = []

    for bubble in bubbles:
        dirichletGridFunction = bempp.api.GridFunction(
            bubble.functionSpace, fun=dirichletFunction
        )
        dirichletProyections = dirichletGridFunction.projections()
        zeros = [0 for proyection in dirichletProyections]
        proyections.append(dirichletProyections)
        proyections.append(zeros)

    rightHandSide = np.concatenate([proyection for proyection in proyections])

    return rightHandSide
