from packages.physicalObjects.physicalValues import *
from .DirichletFormulation import getDirichletFormulation
from .NeumannFormulation import getNeumannFormulation
from .PMCHWTFormulation import getPMCHWTFormulation
from .MullerFormulation import getMullerFormulation


def getFormulation(formulationType, bubbles, logs):
    t0 = tm.time()
    inverseMassMatrix = None
    if formulationType == 0:
        formulation = getDirichletFormulation(bubbles)
    elif formulationType == 1:
        formulation = getNeumannFormulation(bubbles)
    elif formulationType == 2:
        formulation, inverseMassMatrix = getPMCHWTFormulation(bubbles)
    elif formulationType == 3:
        formulation = getMullerFormulation(bubbles)
    else:
        print("Invalid formulation")

    t1 = tm.time()
    if logs:
        print("Formulation:{}{}s".format(" " * 11, round(t1 - t0, 3)))

    return formulation, inverseMassMatrix