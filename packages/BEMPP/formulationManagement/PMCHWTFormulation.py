from packages.physicalObjects.physicalValues import *

class Formulation:
    def __init__(self, numberOfBubbles):
        self.interactionBlocksIds = [[None for _ in range(numberOfBubbles)] for _ in range(numberOfBubbles)]
        self.interactionBlocksById = {}

        self.currentBlock = [[None for _ in range(2)] for _ in range(2)]
        self.mirrorCurrentBlock = [[None for _ in range(2)] for _ in range(2)]

class InteractionBlockId:
    def __init__(self, domainBubble, rangeBubble):
        self.domainSize = domainBubble.size
        self.rangeSize = rangeBubble.size

        usesIntCoordinates = domainBubble.usesIntCoordinates and rangeBubble.usesIntCoordinates 
        if usesIntCoordinates:
            self.index = domainBubble.intCoordinates - rangeBubble.intCoordinates
        else:
            self.index = domainBubble.coordinates - rangeBubble.coordinates

    def __str__(self):
        return str(self.domainSize) + "*" + str(self.rangeSize) + "*" + str(self.index)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, obj):
        isEqual = self.domainSize == obj.domainSize
        isEqual = isEqual and self.rangeSize == obj.rangeSize
        isEqual = isEqual and (self.index == obj.index).all()
        return isEqual

class InverseMassMatrix:
    def __init__(self):
        self.partialInverseMassMatrices = []

def getPMCHWTFormulation(bubbles):
    numberOfBubbles = len(bubbles)
    formulation = Formulation(numberOfBubbles)

    for domainBubbleIndex in range(numberOfBubbles):
        for rangeBubbleIndex in range(domainBubbleIndex, numberOfBubbles):
            decideBubblesInteractionCalculations(
                formulation, bubbles, domainBubbleIndex, rangeBubbleIndex
            )

    inverseMassMatrix = InverseMassMatrix()
    for bubble in bubbles:
        inverseMassMatrix.partialInverseMassMatrices.append(bubble.functionSpace.inverse_mass_matrix())

    return formulation, inverseMassMatrix

def decideBubblesInteractionCalculations(formulation, bubbles, domainBubbleIndex, rangeBubbleIndex):
    domainBubble = bubbles[domainBubbleIndex]
    rangeBubble = bubbles[rangeBubbleIndex]
    
    interactionBlockId = InteractionBlockId(domainBubble, rangeBubble)
    mirrorInteractionBlockId = InteractionBlockId(rangeBubble, domainBubble)

    if interactionBlockId in formulation.interactionBlocksById:
        pass
    else:
        setBubblesInteraction(formulation, domainBubble, rangeBubble, interactionBlockId, mirrorInteractionBlockId)

    formulation.interactionBlocksIds[rangeBubbleIndex][domainBubbleIndex] = interactionBlockId
    if domainBubble != rangeBubble:
        formulation.interactionBlocksIds[domainBubbleIndex][rangeBubbleIndex] = mirrorInteractionBlockId

def setBubblesInteraction(formulation, domainBubble, rangeBubble, interactionBlockId, mirrorInteractionBlockId):
        setBubblesExteriorInteraction(
            formulation, domainBubble, rangeBubble
        )

        if domainBubble == rangeBubble:
            setBubblesInteriorInteraction(formulation, domainBubble)

        setOperatorAsMatrixBlocs(formulation, domainBubble, rangeBubble)

        setBubblesInteractionMatrix(formulation, interactionBlockId, mirrorInteractionBlockId)

def setBubblesExteriorInteraction(
    formulation, domainBubble, rangeBubble
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

    formulation.currentBlock[0][1] = V
    formulation.currentBlock[1][0] = D
    formulation.currentBlock[0][0] = -K

    if domainBubble != rangeBubble:
        T = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
            domainBubble.functionSpace,
            rangeBubble.functionSpace,
            rangeBubble.functionSpace,
            domainBubble.exteriorWavenumber,
        )
        formulation.currentBlock[1][1] = T

def setBubblesInteractionMatrix(formulation, interactionBlockId, mirrorInteractionBlockId):
        currentBlockRows = [
            np.concatenate(
                [formulation.currentBlock[rowIndex][columnIndex] for columnIndex in range(2)],
                axis=1,
            )
            for rowIndex in range(2)
        ]

        currentBlockMatrix = np.concatenate([row for row in currentBlockRows])
        del currentBlockRows
        formulation.interactionBlocksById[interactionBlockId] = currentBlockMatrix
        del currentBlockMatrix

        if interactionBlockId == mirrorInteractionBlockId:
            return
        
        mirrorCurrentBlockRows = [
            np.concatenate(
                [formulation.mirrorCurrentBlock[rowIndex][columnIndex] for columnIndex in range(2)],
                axis=1,
            )
            for rowIndex in range(2)
        ]
        mirrorCurrentBlockMatrix = np.concatenate([row for row in mirrorCurrentBlockRows])
        del mirrorCurrentBlockRows
        formulation.interactionBlocksById[mirrorInteractionBlockId] = mirrorCurrentBlockMatrix
        del mirrorCurrentBlockMatrix

def setBubblesInteriorInteraction(formulation, bubble):
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
    K = bempp.api.operators.boundary.helmholtz.double_layer(
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.functionSpace,
        bubble.interiorWavenumber,
    )

    formulation.currentBlock[0][1] += (sigmaAmb / sigmaInt) * V
    formulation.currentBlock[1][0] += (sigmaInt / sigmaAmb) * D
    formulation.currentBlock[0][0] += -K

def setOperatorAsMatrixBlocs(formulation, domainBubble, rangeBubble):
    formulation.currentBlock[0][1] = bempp.api.as_matrix(
        formulation.currentBlock[0][1].weak_form()
    )
    formulation.currentBlock[1][0] = bempp.api.as_matrix(
        formulation.currentBlock[1][0].weak_form()
    )

    formulation.currentBlock[0][0] = bempp.api.as_matrix(
        formulation.currentBlock[0][0].weak_form()
    )

    if domainBubble == rangeBubble:
        formulation.currentBlock[1][1] = -formulation.currentBlock[0][0].T
        return
    formulation.currentBlock[1][1] = bempp.api.as_matrix(
        formulation.currentBlock[1][1].weak_form()
    )
    
    
    
    formulation.mirrorCurrentBlock[1][0] = formulation.currentBlock[1][0].T
    formulation.mirrorCurrentBlock[0][1] = formulation.currentBlock[0][1].T

    formulation.mirrorCurrentBlock[0][0] = -formulation.currentBlock[1][1].T
    formulation.mirrorCurrentBlock[1][1] = -formulation.currentBlock[0][0].T
