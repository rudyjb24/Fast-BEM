from packages.physicalObjects.physicalValues import *


def getMullerFormulation(bubbles):
    numberOfBubbles = len(bubbles)
    formulationSize = 2 * numberOfBubbles
    formulation = [[0 for _ in range(formulationSize)] for _ in range(formulationSize)]

    for domainBubbleIndex in range(numberOfBubbles):
        for rangeBubbleIndex in range(domainBubbleIndex, numberOfBubbles):
            setBubblesInteraction(
                formulation, bubbles, domainBubbleIndex, rangeBubbleIndex
            )

    return formulation


def setBubblesInteraction(formulation, bubbles, domainBubbleIndex, rangeBubbleIndex):
    domainBubble = bubbles[domainBubbleIndex]
    rangeBubble = bubbles[rangeBubbleIndex]

    baseRow = 2 * rangeBubbleIndex
    baseColumn = 2 * domainBubbleIndex

    setBubblesExteriorInteraction(
        formulation, domainBubble, rangeBubble, baseRow, baseColumn
    )

    if domainBubbleIndex == rangeBubbleIndex:
        setBubblesInteriorInteraction(formulation, domainBubble, baseRow, baseColumn)

    setOperatorAsMatrixBlocs(formulation, baseRow, baseColumn)


def setBubblesExteriorInteraction(
    formulation, domainBubble, rangeBubble, baseRow, baseColumn
):
    V = bempp.api.operators.boundary.helmholtz.single_layer(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )
    D = bempp.api.operators.boundary.helmholtz.hypersingular(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )
    K = bempp.api.operators.boundary.helmholtz.double_layer(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )
    T = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )

    formulation[baseRow][baseColumn + 1] = V
    formulation[baseRow + 1][baseColumn] = D
    formulation[baseRow][baseColumn] = -K
    formulation[baseRow + 1][baseColumn + 1] = T


def setBubblesInteriorInteraction(formulation, bubble, baseRow, baseColumn):
    V = bempp.api.operators.boundary.helmholtz.single_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )
    D = bempp.api.operators.boundary.helmholtz.hypersingular(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )
    I = bempp.api.operators.boundary.sparse.identity(
        bubble.functionSpace, bubble.functionSpace, bubble.functionSpace
    )
    K = bempp.api.operators.boundary.helmholtz.double_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )
    T = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )

    formulation[baseRow][baseColumn + 1] += -(sigmaAmb / sigmaInt) * V
    formulation[baseRow + 1][baseColumn] += -(sigmaInt / sigmaAmb) * D
    formulation[baseRow][baseColumn] += I + K
    formulation[baseRow + 1][baseColumn + 1] += I - T


def setOperatorAsMatrixBlocs(formulation, baseRow, baseColumn):
    formulation[baseRow][baseColumn + 1] = bempp.api.as_matrix(
        formulation[baseRow][baseColumn + 1].weak_form()
    )
    formulation[baseRow + 1][baseColumn] = bempp.api.as_matrix(
        formulation[baseRow + 1][baseColumn].weak_form()
    )

    formulation[baseRow][baseColumn] = bempp.api.as_matrix(
        formulation[baseRow][baseColumn].weak_form()
    )
    formulation[baseRow + 1][baseColumn + 1] = bempp.api.as_matrix(
        formulation[baseRow + 1][baseColumn + 1].weak_form()
    )

    if baseRow == baseColumn:
        return

    formulation[baseColumn + 1][baseRow] = formulation[baseRow + 1][baseColumn].T
    formulation[baseColumn][baseRow + 1] = formulation[baseRow][baseColumn + 1].T

    formulation[baseColumn][baseRow] = -formulation[baseRow + 1][baseColumn + 1].T
    formulation[baseColumn + 1][baseRow + 1] = -formulation[baseRow][baseColumn].T
