from packages.physicalObjects.physicalValues import *
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from .formulationManagement.formulationGenerator import getFormulation
from .rightHandSideManagement.rightHandSideGenerator import getRightHandSide
from packages.BEMPP.gridVariables import GridVariables
from packages.physicalObjects.bubble import Bubble

it_count = 0
def iteration_counter(x):
    global it_count
    it_count += 1

formulation = None
inverseMassMatrix = None
def customMatVec(x):
    global formulation
    global inverseMassMatrix

    splitedVector = splitVector(formulation, x)
    result = [np.zeros(partialVector.shape, dtype=np.complex128) for partialVector in splitedVector]

    for rowIndex, rowsInteractionBlocksIds in enumerate(formulation.interactionBlocksIds):
    
        for columnIndex, interactionBlockId in enumerate(rowsInteractionBlocksIds):
            interactionBlockMatrix = formulation.interactionBlocksById[interactionBlockId]
            result[rowIndex] += np.dot(interactionBlockMatrix, splitedVector[columnIndex])

        partialInverseMassMatrix = inverseMassMatrix.partialInverseMassMatrices[rowIndex]
        size = result[rowIndex].size
        halfSize = size//2
        firstHalf = result[rowIndex][:halfSize]
        secondHalf = result[rowIndex][halfSize:]

        firstHalf = partialInverseMassMatrix * firstHalf
        secondHalf = partialInverseMassMatrix * secondHalf
        
        result[rowIndex] = np.concatenate((firstHalf, secondHalf))

    return np.concatenate(result)

def splitVector(formulation, x):
    splitedVector = []
    lowerIndex = 0
    higherIndex = 0
    for interactionBlockId in formulation.interactionBlocksIds[0]:
        interactionBlockMatrix = formulation.interactionBlocksById[interactionBlockId]

        numberOfColumns = interactionBlockMatrix.shape[1]
        higherIndex = lowerIndex + numberOfColumns
        
        splitedVector.append(x[lowerIndex:higherIndex])

        lowerIndex = higherIndex

    return splitedVector

class SingleTraceManager:
    def __init__(self, formulationType=2, logs=False):
        self.experimentId = "{}".format(formulationType)
        
        self.logs = logs
        self.formulationType = formulationType

        self.frequency = None
        self.exteriorWavenumber = None

        self.bubbles = []

        self.formulation = None
        self.inverseMassMatrix = None
        self.rightHandSide = None

        self.gmresSolution = None
        self.dirichletTotalFields = []
        self.neumannTotalFields = []

        self.exteriorPotentialsV = []
        self.exteriorPotentialsK = []

        self.interiorPotentialV = None
        self.interiorPotentialK = None

    def setNewBubble(
        self, center=(0, 0, 0), size=1, isSquare=False, kind="P", degree=1, h=0.3, intCoordinates=None
    ):
        gridVariables = GridVariables(kind, degree, h)
        baseBubble = self.getBaseBubble(size, gridVariables)
        if len(self.bubbles) > 1:
            newBubble = Bubble(center, size, isSquare, gridVariables, baseBubble, False, intCoordinates)
        else:
            newBubble = Bubble(center, size, isSquare, gridVariables, baseBubble, self.logs, intCoordinates)
        self.bubbles.append(newBubble)
        self.updateExperimentId(newBubble)

    def getBaseBubble(self, size, gridVariables):
        for bubble in self.bubbles:
            if bubble.size != size:
                continue
            elif bubble.gridVariables.kind != gridVariables.kind:
                continue
            elif bubble.gridVariables.degree != gridVariables.degree:
                continue
            elif bubble.gridVariables.h != gridVariables.h:
                continue

            return bubble
        return None

    def updateExperimentId(self, newBubble):
        self.experimentId += "&{}".format(newBubble.size)

        if newBubble.usesIntCoordinates:
            coordinates = newBubble.intCoordinates
        else:
            coordinates = newBubble.coordinates

        self.experimentId += "*{}".format(coordinates)

    def calculateFieldsFromMultipleFrequencies(
        self,
        scatteringPoint,
        frequencyRange=None,
        wavenumberRange=None,
        numberOfPoints=1,
        refineSweepSample=False,
        spacing_multiplier=None
    ):
        frequencies, wavenumbers = self.checkPickleFile(frequencyRange, wavenumberRange, numberOfPoints, spacing_multiplier)
        self.logs = len(frequencies) <= 5

        incidentFields = np.zeros(frequencies.size, dtype=np.complex128)
        scatteredFields = np.zeros(frequencies.size, dtype=np.complex128)
        totalFields = np.zeros(frequencies.size, dtype=np.complex128)
        for i in range(frequencies.size):
            self.setCalculationParameters(frequencies[i])
            (
                incidentFields[i],
                scatteredFields[i],
                totalFields[i],
            ) = self.evaluateExteriorFieldOn(scatteringPoint)

            self.saveNewDataPoint(frequencies[i], wavenumbers[i], incidentFields[i], scatteredFields[i], totalFields[i])

        scatteringPoint = np.array(
            (scatteringPoint[0][0], scatteringPoint[1][0], scatteringPoint[2][0])
        )
        scatteringDistance = np.linalg.norm(
            self.bubbles[0].coordinates - scatteringPoint
        )
        if (scatteringDistance) > 500 * self.bubbles[0].size:
            scatteredFields = scatteringDistance * scatteredFields
            totalFields = scatteredFields + incidentFields

        return incidentFields, scatteredFields, totalFields, frequencies, wavenumbers

    def checkPickleFile(self, frequencyRange, wavenumberRange, numberOfPoints, spacing_multiplier):
        self.spacing_multiplier = spacing_multiplier
        fileExisted = self.loadSavedData()

        if not fileExisted:
            return self.defaultFrequenciesAndWavenumbers(frequencyRange, wavenumberRange, numberOfPoints)


        if frequencyRange == None:
            minFrequency = wavenumberRange[0] * v / (2*pi)
            maxFrequency = wavenumberRange[1] * v / (2*pi)
        else:
            minFrequency, maxFrequency = frequencyRange

        allFrequencies = list(self.pickle['frequencies'])

        newFrequencies = list()
        newWavenumbers = list()

        finished = False
        tol = 0
        remainingPoints = numberOfPoints


        while not finished:
            finished = True
            apendableFrequencies = list()
            for index, frequency in enumerate(allFrequencies):
                if remainingPoints == 0:
                    continue

                if index == 0:
                    if abs(frequency - minFrequency)/minFrequency <= tol:
                        continue
                    newFrequency = (frequency + minFrequency) / 2
                elif index == len(allFrequencies) - 1:
                    if abs(frequency - maxFrequency)/frequency <= tol:
                        continue
                    newFrequency = (frequency + maxFrequency) / 2
                else:
                    if abs(frequency - allFrequencies[index+1])/frequency <= tol:
                        continue
                    newFrequency = (frequency + allFrequencies[index+1]) / 2

                newWavenumber = newFrequency * (2*pi) / v
                
                if (newFrequency < minFrequency) or (newFrequency > maxFrequency) or (newFrequency in newFrequencies):
                    continue

                newFrequencies.append(newFrequency)
                newWavenumbers.append(newWavenumber)


                apendableFrequencies.append((index + 1, newFrequency))
                remainingPoints -= 1
                finished = False

            for delta, stuf in enumerate(apendableFrequencies):
                allFrequencies.insert(stuf[0]+delta, stuf[1])

            finished = finished or remainingPoints == 0
        return np.array(newFrequencies), np.array(newWavenumbers)

    def defaultFrequenciesAndWavenumbers(self, frequencyRange, wavenumberRange, numberOfPoints):
        if frequencyRange == None:
            wavenumbers = np.linspace(wavenumberRange[0], wavenumberRange[1], numberOfPoints)
            frequencies = wavenumbers * v / (2*pi)
        elif wavenumberRange == None:
            frequencies = np.linspace(frequencyRange[0], frequencyRange[1], numberOfPoints)
            wavenumbers = frequencies * (2*pi) / v

        return frequencies, wavenumbers

    def loadSavedData(self):
        import os
        pickleFolder = "pickles/L={}*radius".format(self.spacing_multiplier)
        if not os.path.isdir(pickleFolder):
            os.mkdir(pickleFolder)
        self.picklePath = pickleFolder+"/{}${}".format(self.formulationType, len(self.bubbles))
        try:
            with open(self.picklePath, "rb") as file:
                self.pickle = pickle.load(file)
            fileExisted = True
        except:
            self.pickle = {'frequencies': [], 'wavenumbers': [], 'incidentFields': [], 'scatteredFields': [],'totalFields': []}
            with open(self.picklePath, "wb") as file:
                pickle.dump(self.pickle, file)
            fileExisted = False
        
        return fileExisted

    def saveNewDataPoint(self, frequency, wavenumber, incidentField, scatteredField, totalField):
        if frequency in self.pickle['frequencies']:
            return

        index = 0
        if len(self.pickle['frequencies']) > 0 and frequency > self.pickle['frequencies'][0]:
        
            for comparationFrequency in self.pickle['frequencies']: # should do a binary search
                if frequency < comparationFrequency:
                    break
                index += 1

        self.pickle['frequencies'].insert(index, frequency)
        self.pickle['wavenumbers'].insert(index, wavenumber)
        self.pickle['incidentFields'].insert(index, incidentField)
        self.pickle['scatteredFields'].insert(index, scatteredField)
        self.pickle['totalFields'].insert(index, totalField)

        with open(self.picklePath, "wb") as file:
            pickle.dump(self.pickle, file)

    def setCalculationParameters(self, newFrequency):
        self.setWavenumbers(newFrequency)
        self.formulation, self.inverseMassMatrix = getFormulation(self.formulationType, self.bubbles, self.logs)
        self.rightHandSide = getRightHandSide(
            self.formulationType, self.bubbles, self.logs
        )
        self.executeGmres()
        self.setDirichletAndNeumannTotalFields()

    def setWavenumbers(self, newFrequency):
        self.frequency = newFrequency
        self.exteriorWavenumber = 2 * pi * self.frequency / v
        for bubble in self.bubbles:
            bubble.setWavenumbers(
                self.frequency,
                self.exteriorWavenumber,
            )

    def executeGmres(self):
        t0 = tm.time()
        global it_count
        it_count = 0

        global formulation
        formulation = self.formulation
        global inverseMassMatrix
        inverseMassMatrix = self.inverseMassMatrix
        customLinearOperator = LinearOperator((self.rightHandSide.size, self.rightHandSide.size), matvec = customMatVec)

        rightHandSideNorm = np.linalg.norm(self.rightHandSide)

        self.gmresSolution, info = gmres(
            customLinearOperator,
            self.rightHandSide,
            callback=iteration_counter,
            maxiter=1000,
            restart=1000,
            tol=1e-5,
        )
        if info:
            if info > 0 and self.logs:
                print("Gmres did not converge")
        rightHandSideAproximation = customMatVec(self.gmresSolution)
        gmresError = np.linalg.norm(
                rightHandSideAproximation - self.rightHandSide
            )
        t1 = tm.time()
        if self.logs:
            print(
                "Gmres:{}{}s - {} iterations - {} error - RHS norm: {}".format(
                    " " * 17, round(t1 - t0, 3), it_count, gmresError, rightHandSideNorm
                )
            )

    def setDirichletAndNeumannTotalFields(self):
        t0 = tm.time()
        self.dirichletTotalFields = []
        self.neumannTotalFields = []
        startIndex = 0
        for bubble in self.bubbles:
            middleIndex = startIndex + bubble.functionSpace.global_dof_count
            endIndex = startIndex + 2 * bubble.functionSpace.global_dof_count

            dirichletTotalField = bempp.api.GridFunction(
                bubble.functionSpace,
                coefficients=self.gmresSolution[startIndex:middleIndex],
            )
            self.dirichletTotalFields.append(dirichletTotalField)

            neumannTotalField = bempp.api.GridFunction(
                bubble.functionSpace,
                coefficients=self.gmresSolution[middleIndex:endIndex],
            )
            self.neumannTotalFields.append(neumannTotalField)

            startIndex = endIndex
        t1 = tm.time()
        if self.logs:
            print("Total Fields:{}{}s".format(" " * 10, round(t1 - t0, 3)))

    def evaluateExteriorFieldOn(self, exteriorPoints):
        t0 = tm.time()

        self.setExteriorPotentials(exteriorPoints)

        scatteredExteriorField = (
            self.exteriorPotentialsK[0] * self.dirichletTotalFields[0]
            - self.exteriorPotentialsV[0] * self.neumannTotalFields[0]
        )
        for i in range(1, len(self.bubbles)):
            scatteredExteriorField += (
                self.exteriorPotentialsK[i] * self.dirichletTotalFields[i]
                - self.exteriorPotentialsV[i] * self.neumannTotalFields[i]
            )
        scatteredExteriorField = (scatteredExteriorField).ravel()

        incidentExteriorField = self.incidentField(exteriorPoints)
        totalExteriorField = scatteredExteriorField + incidentExteriorField

        t1 = tm.time()
        if self.logs:
            print(
                "Exterior calculations: {}s (for {} points)".format(
                    round(t1 - t0, 3), int(exteriorPoints.size/3)
                )
            )

        return incidentExteriorField, scatteredExteriorField, totalExteriorField

    def setExteriorPotentials(self, exteriorPoints):
        t0 = tm.time()
        self.exteriorPotentialsV = []
        self.exteriorPotentialsK = []
        for bubble in self.bubbles:
            exteriorPotentialV = bempp.api.operators.potential.helmholtz.single_layer(
                bubble.functionSpace, exteriorPoints, self.exteriorWavenumber
            )
            self.exteriorPotentialsV.append(exteriorPotentialV)

            exteriorPotentialK = bempp.api.operators.potential.helmholtz.double_layer(
                bubble.functionSpace, exteriorPoints, self.exteriorWavenumber
            )
            self.exteriorPotentialsK.append(exteriorPotentialK)

        t1 = tm.time()
        if self.logs:
            print("Exterior potential:{}{}s".format(" " * 4, round(t1 - t0, 3)))

    def calculateFieldsFromMultipleFrequenciesMultiplePoints(
        self, referencePoint, scatteringPoint, frequencies
    ):
        self.logs = False

        incidentFields = np.zeros(frequencies.size, dtype=np.complex128)
        scatteredFields = np.zeros(frequencies.size, dtype=np.complex128)

        for frequencyIndex in range(frequencies.size):
            self.setCalculationParameters(frequencies[frequencyIndex])
            incidentFields[frequencyIndex] = self.incidentField(referencePoint)
            a, scatteredFields[frequencyIndex], b = self.evaluateExteriorFieldOn(
                scatteringPoint
            )

        return incidentFields, scatteredFields

    def calculateSpatialFields(self, newFrequency, points, pointsDistribution, spacing_multiplier=0):
        self.setCalculationParameters(newFrequency)
        return self.evaluateFieldOn(points, pointsDistribution)

    def loadTotalFields(self, frequency, spacing_multiplier):
            if spacing_multiplier == 0:
                return False
            self.spacing_multiplier = spacing_multiplier
            self.loadSavedDataMSH()

            if 'totalFieldFunctions' not in self.pickle:
                return False

            if frequency not in self.pickle['totalFieldFunctions']:
                return False

            self.dirichletTotalField = self.pickle['totalFieldFunctions'][frequency]['dirichlet']
            self.neumannTotalFields = self.pickle['totalFieldFunctions'][frequency]['neumann']

            return True

    def loadSavedDataMSH(self):
        self.mshPath = "msh/L={}*radius/{}${}".format(self.spacing_multiplier, self.formulationType, len(self.bubbles))
        try:
            with open(self.picklePath, "rb") as file:
                self.pickle = pickle.load(file)
            fileExisted = True
        except:
            self.pickle = {'frequencies': [], 'wavenumbers': [], 'incidentFields': [], 'scatteredFields': [],'totalFields': []}
            with open(self.picklePath, "wb") as file:
                pickle.dump(self.pickle, file)
            fileExisted = False
        
        return fileExisted

    def saveTotalFields(self, frequency):
            if 'totalFieldFunctions' not in self.pickle:
                self.pickle['totalFieldFunctions'] = {}

            self.pickle['totalFieldFunctions'][frequency] = {}
            self.pickle['totalFieldFunctions'][frequency]['dirichlet'] = self.dirichletTotalFields
            self.pickle['totalFieldFunctions'][frequency]['neumann'] = self.neumannTotalFields

            with open(self.picklePath, "wb") as file:
                pickle.dump(self.pickle, file)

    def evaluateFieldOn(self, points, pointsDistribution):
        scatteredField = np.zeros(points.shape[1], dtype="complex128")
        incidentField = np.zeros(points.shape[1], dtype="complex128")
        totalField = np.zeros(points.shape[1], dtype="complex128")

        (
            exteriorPoints,
            exteriorIndexes,
            interiorPoints,
            interiorIndexes,
        ) = self.separateExteriorAndInteriorPoints(points)

        (
            incidentExteriorField,
            scatteredExteriorField,
            totalExteriorField,
        ) = self.evaluateExteriorFieldOn(exteriorPoints)
        scatteredField[exteriorIndexes] = scatteredExteriorField
        incidentField[exteriorIndexes] = incidentExteriorField
        totalField[exteriorIndexes] = totalExteriorField

        for i in range(len(self.bubbles)):
            (
                incidentInteriorField,
                scatteredInteriorField,
                totalInteriorField,
            ) = self.evaluateInteriorFieldOn(interiorPoints[i], i)
            scatteredField[interiorIndexes[i]] = scatteredInteriorField
            incidentField[interiorIndexes[i]] = incidentInteriorField
            totalField[interiorIndexes[i]] = totalInteriorField

        scatteredField = scatteredField.reshape(pointsDistribution)
        incidentField = incidentField.reshape(pointsDistribution)
        totalField = totalField.reshape(pointsDistribution)

        return incidentField, scatteredField, totalField

    def separateExteriorAndInteriorPoints(self, points):
        xCoordinates, yCoordinates = points[:2]

        exteriorIndexes = np.repeat(True, xCoordinates.size)
        interiorIndexes = []
        for bubble in self.bubbles:
            currentInteriorIndexes = (xCoordinates-bubble.coordinates[1])**2 + (yCoordinates-bubble.coordinates[2])**2 <= bubble.size**2

            interiorIndexes.append(currentInteriorIndexes)
            exteriorIndexes = np.logical_and(
                exteriorIndexes, np.logical_not(currentInteriorIndexes)
            )

        exteriorPoints = points[:, exteriorIndexes]
        interiorPoints = [
            points[:, currentInteriorIndexes]
            for currentInteriorIndexes in interiorIndexes
        ]

        return exteriorPoints, exteriorIndexes, interiorPoints, interiorIndexes

    def evaluateInteriorFieldOn(self, interiorPoints, bubbleIndex):
        t0 = tm.time()

        self.setInteriorPotentials(interiorPoints, bubbleIndex)
        totalInteriorField = (
            (sigmaAmb / sigmaInt)
            * self.interiorPotentialV
            * self.neumannTotalFields[bubbleIndex]
            - self.interiorPotentialK * self.dirichletTotalFields[bubbleIndex]
        ).ravel()
        incidentInteriorField = self.incidentField(interiorPoints)
        scatteredInteriorField = totalInteriorField - incidentInteriorField

        t1 = tm.time()
        if self.logs:
            print(
                "Interior calculations: {}s (for {} points)".format(
                    round(t1 - t0, 3), interiorPoints.size
                )
            )

        return incidentInteriorField, scatteredInteriorField, totalInteriorField

    def setInteriorPotentials(self, interiorPoints, bubbleIndex):
        t0 = tm.time()

        bubble = self.bubbles[bubbleIndex]
        self.interiorPotentialV = bempp.api.operators.potential.helmholtz.single_layer(
            bubble.functionSpace, interiorPoints, bubble.interiorWavenumber
        )

        self.interiorPotentialK = bempp.api.operators.potential.helmholtz.double_layer(
            bubble.functionSpace, interiorPoints, bubble.interiorWavenumber
        )

        t1 = tm.time()
        if self.logs:
            print("Interior potential:{}{}s".format(" " * 4, round(t1 - t0, 3)))

    def incidentField(self, points):
        return np.exp(1j * self.exteriorWavenumber * points[0])
