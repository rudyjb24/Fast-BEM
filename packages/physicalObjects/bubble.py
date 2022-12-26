from packages.physicalObjects.physicalValues import *


class Bubble:
    def __init__(
        self, coordinates, size, isSquare=False, gridVariables=None, baseBubble=None, logs=False, intCoordinates=None
    ):

        self.coordinates = np.array(coordinates)
        self.usesIntCoordinates = intCoordinates != None
        if self.usesIntCoordinates:
            self.intCoordinates = np.array(intCoordinates)
        else:
            self.intCoordinates = intCoordinates
        self.size = size
        self.gridVariables = gridVariables
        self.isSquare = isSquare
        self.logs = logs

        self.m_equivalentMass = rho / (4 * pi * self.size)
        self.kappa_stiffness = (3 * varphi * PA) / (4 * pi * self.size ** 3)
        self.b_damping = self.kappa_stiffness * self.size / v

        self.grid = self.getGrid(baseBubble)
        self.functionSpace = self.getFunctionSpace()

        self.frequency = None

        self.exteriorWavenumber = None

        self.interiorSoundSpeed = vInt
        self.interiorWavenumber = None
        self.interiorDensity = rhoInt

    def getGrid(self, baseBubble):
        if not self.gridVariables:
            return

        t0 = tm.time()

        if not baseBubble:
            grid = self.getGridFromScratch()
        else:
            grid = self.getGridFromTranslation(baseBubble)

        t1 = tm.time()
        if self.logs:
            print("Grid:{}{}s".format(" " * 13, round(t1 - t0, 3)))

        return grid

    def getGridFromScratch(self):
        if self.isSquare:
            grid = bempp.api.shapes.cube(
                self.size, self.coordinates, self.gridVariables.h
            )
        else:
            grid = bempp.api.shapes.sphere(
                self.size, self.coordinates, self.gridVariables.h
            )
        return grid

    def getGridFromTranslation(self, baseBubble):
        vertices = baseBubble.grid.vertices
        numberOfVertices = vertices.shape[1]
        
        translation = self.coordinates - baseBubble.coordinates
        translationMatrix = np.array([translationValue * np.ones(numberOfVertices) for translationValue in translation])

        newVertices = vertices + translationMatrix

        return bempp.api.Grid(newVertices, baseBubble.grid.elements)

    def getFunctionSpace(self):
        if not self.gridVariables:
            return

        t0 = tm.time()

        functionSpace = bempp.api.function_space(
            self.grid, self.gridVariables.kind, self.gridVariables.degree
        )

        t1 = tm.time()
        if self.logs:
            print("Function space:{}{}s".format(" " * 5, round(t1 - t0, 3)))

        return functionSpace

    def setWavenumbers(
        self,
        newfrequency,
        newExteriorWavenumber,
    ):
        self.frequency = newfrequency

        self.exteriorWavenumber = newExteriorWavenumber

        self.interiorWavenumber = 2 * pi * self.frequency / vInt
