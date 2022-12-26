import numpy as np
import time as tm
import bempp.api
import pickle

# Incident wave
vInc_incidentWaveDirection = np.array((1, 0, 0))
vInc = vInc_incidentWaveDirection

PInc_incidentPressureAmplitude_Pa = 1
PInc = PInc_incidentPressureAmplitude_Pa


def calculateIncidentField(x, k):
    return np.exp(i * k * np.dot(x, vInc))


def calculateIncidentFieldsFromMultipleFrequencies(
    sampleSize, wavenumbers, scatteringPoint
):
    incidentFields = np.zeros(sampleSize, dtype=np.complex128)

    for index, wavenumber in enumerate(wavenumbers):
        incidentField = calculateIncidentField(scatteringPoint, wavenumber)
        incidentFields[index] = incidentField

    return incidentFields


# Ambient physical values
PA_ambienPressure_Pa = 1.01e5
PA = PA_ambienPressure_Pa

v_ambientWaveSpeed_meterPerSecond = 1480
v = v_ambientWaveSpeed_meterPerSecond

rho_ambientDensity_kgPerMeterCubed = 997
rho = rho_ambientDensity_kgPerMeterCubed

sigma_ambientMaterialConstant_MeterCubedPerkg = 1 / rho
sigmaAmb = sigma_ambientMaterialConstant_MeterCubedPerkg

# Interior physical values
v_interiorWaveSpeed_meterPerSecond = 343
vInt = v_interiorWaveSpeed_meterPerSecond

rho_interiorDensity_kgPerMeterCubed = 1.225
rhoInt = rho_interiorDensity_kgPerMeterCubed

sigma_interiorMaterialConstant_MeterCubedPerkg = 1 / rhoInt
sigmaInt = sigma_interiorMaterialConstant_MeterCubedPerkg

# Time
t_time_second = 0
t = 0

# Constants
varphi_specificAirRatio = 1.4
varphi = varphi_specificAirRatio
i = complex(0, 1)
pi = np.pi
