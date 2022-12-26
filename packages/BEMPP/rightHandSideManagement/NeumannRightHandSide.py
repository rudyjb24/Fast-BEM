from packages.physicalObjects.physicalValues import *


def getNeumannRightHandSide(bubbles, neumannFunction):
    proyections = []

    for bubble in bubbles:
        neumannGridFunction = bempp.api.GridFunction(
            bubble.functionSpace, fun=neumannFunction
        )
        neumannProyections = neumannGridFunction.projections()
        zeros = [0 for proyection in neumannProyections]
        proyections.append(neumannProyections)
        proyections.append(zeros)

    rightHandSide = np.concatenate([proyection for proyection in proyections])

    return rightHandSide
