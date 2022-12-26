from packages.physicalObjects.physicalValues import *


def getNeumannFormulation(bubbles):
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
    D = bempp.api.operators.boundary.helmholtz.hypersingular(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )
    T1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        domainBubble.functionSpace,
        rangeBubble.functionSpace,
        rangeBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )

    formulation[baseRow][baseColumn] = D
    formulation[baseRow][baseColumn + 1] = T1

    if domainBubble == rangeBubble:
        return

    T2 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        rangeBubble.functionSpace,
        domainBubble.functionSpace,
        domainBubble.functionSpace,
        domainBubble.exteriorWavenumber,
    )
    formulation[baseColumn][baseRow + 1] = T2


def setBubblesInteriorInteraction(formulation, bubble, baseRow, baseColumn):
    D = bempp.api.operators.boundary.helmholtz.hypersingular(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )
    I = bempp.api.operators.boundary.sparse.identity(
        bubble.functionSpace, bubble.functionSpace, bubble.functionSpace
    )
    T = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )

    formulation[baseRow + 1][baseColumn] = (sigmaInt / sigmaAmb) * D
    formulation[baseRow][baseColumn + 1] += (1 / 2) * I
    formulation[baseRow + 1][baseColumn + 1] = -(1 / 2) * I + T


def setOperatorAsMatrixBlocs(formulation, baseRow, baseColumn):
    formulation[baseRow][baseColumn] = bempp.api.as_matrix(
        formulation[baseRow][baseColumn].weak_form()
    )
    formulation[baseRow][baseColumn + 1] = bempp.api.as_matrix(
        formulation[baseRow][baseColumn + 1].weak_form()
    )

    if baseRow == baseColumn:
        formulation[baseRow + 1][baseColumn] = bempp.api.as_matrix(
            formulation[baseRow + 1][baseColumn].weak_form()
        )
        formulation[baseRow + 1][baseColumn + 1] = bempp.api.as_matrix(
            formulation[baseRow + 1][baseColumn + 1].weak_form()
        )
        return

    formulation[baseRow + 1][baseColumn] = np.zeros(
        formulation[baseRow][baseColumn].shape
    )
    formulation[baseRow + 1][baseColumn + 1] = np.zeros(
        formulation[baseRow][baseColumn + 1].shape
    )

    formulation[baseColumn][baseRow] = -formulation[baseRow + 1][baseColumn + 1].T
    formulation[baseColumn][baseRow + 1] = bempp.api.as_matrix(
        formulation[baseColumn][baseRow + 1].weak_form()
    )

    formulation[baseColumn + 1][baseRow] = formulation[baseRow + 1][baseColumn].T
    formulation[baseColumn + 1][baseRow + 1] = formulation[baseRow + 1][
        baseColumn + 1
    ].T
