from packages.physicalObjects.physicalValues import *


def getDirichletFormulation(bubbles):
    numberOfBubbles = len(bubbles)
    formulationSize = 2 * numberOfBubbles
    formulation = [[0 for _ in range(formulationSize)] for _ in range(formulationSize)]

    for rangeBubbleIndex in range(numberOfBubbles):
        for domainBubbleIndex in range(rangeBubbleIndex, numberOfBubbles):
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
    K1 = bempp.api.operators.boundary.helmholtz.double_layer(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )

    formulation[baseRow][baseColumn + 1] = V
    formulation[baseRow][baseColumn] = -K1

    if domainBubble == rangeBubble:
        return

    K2 = bempp.api.operators.boundary.helmholtz.double_layer(
        rangeBubble.functionSpace,
        domainBubble.functionSpace,
        domainBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )
    formulation[baseColumn][baseRow] = -K2


def setBubblesInteriorInteraction(formulation, bubble, baseRow, baseColumn):
    V = bempp.api.operators.boundary.helmholtz.single_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )
    K = bempp.api.operators.boundary.helmholtz.double_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )
    I = bempp.api.operators.boundary.sparse.identity(
        bubble.functionSpace, bubble.functionSpace, bubble.functionSpace
    )

    formulation[baseRow + 1][baseColumn + 1] = (sigmaAmb / sigmaInt) * V
    formulation[baseRow][baseColumn] += (1 / 2) * I
    formulation[baseRow + 1][baseColumn] = -(1 / 2) * I - K


def setOperatorAsMatrixBlocs(formulation, baseRow, baseColumn):
    formulation[baseRow][baseColumn + 1] = bempp.api.as_matrix(
        formulation[baseRow][baseColumn + 1].weak_form()
    )
    formulation[baseRow][baseColumn] = bempp.api.as_matrix(
        formulation[baseRow][baseColumn].weak_form()
    )

    if baseRow == baseColumn:
        formulation[baseRow + 1][baseColumn] = bempp.api.as_matrix(
            formulation[baseRow + 1][baseColumn].weak_form()
        )
        formulation[baseRow + 1][baseColumn + 1] = bempp.api.as_matrix(
            formulation[baseRow + 1][baseColumn + 1].weak_form()
        )
        return

    numberOfRows = formulation[baseRow][baseRow + 1].shape[0]
    formulation[baseRow + 1][baseColumn] = np.zeros(
        (numberOfRows, formulation[baseRow][baseColumn].shape[1])
    )
    formulation[baseRow + 1][baseColumn + 1] = np.zeros(
        (numberOfRows, formulation[baseRow][baseColumn + 1].shape[1])
    )

    formulation[baseColumn][baseRow + 1] = formulation[baseRow][baseColumn + 1].T
    formulation[baseColumn][baseRow] = bempp.api.as_matrix(
        formulation[baseColumn][baseRow].weak_form()
    )

    formulation[baseColumn + 1][baseRow] = formulation[baseRow + 1][baseColumn].T
    formulation[baseColumn + 1][baseRow + 1] = formulation[baseRow + 1][
        baseColumn + 1
    ].T
