import numpy as np
import matplotlib.pyplot as plt


def getAbsoluteRealImaginaryValues(complexValues):
    absolute = np.abs(complexValues)
    real = np.real(complexValues)
    imaginary = np.imag(complexValues)

    return absolute, real, imaginary


def plotPic(
    wavenumbers,
    frequencies,
    absoluteScatteredFields,
    realScatteredFields=None,
    imaginaryScatteredFields=None,
    expectedNumbersOfPics=1,
    plotVsWavenumbers=True
):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot()
    if plotVsWavenumbers:
        xAxis = wavenumbers
    else:
        xAxis = frequencies
    plt.plot(xAxis, absoluteScatteredFields, color="C4", label="Absolute")
    if realScatteredFields is not None:
        plt.plot(xAxis, realScatteredFields, color="C0", label="Real")
    if imaginaryScatteredFields is not None:
        plt.plot(xAxis, imaginaryScatteredFields, color="C3", label="Imaginary")
    if plotVsWavenumbers:
        ax.set_xlabel("Wavenumber (1/m)")
    else:
        ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude of transmission")
    plt.legend()
    plt.savefig('test.pdf', bbox_inches='tight')
    plt.show()
    printMaximumAndMinimums(
        realScatteredFields,
        absoluteScatteredFields,
        wavenumbers,
        frequencies,
        expectedNumbersOfPics,
    )


def printMaximumAndMinimums(
    realScatteredFields,
    absoluteScatteredFields,
    wavenumbers,
    frequencies,
    expectedNumbersOfPics,
):
    chunkSize = int(wavenumbers.size / expectedNumbersOfPics)
    for i in range(expectedNumbersOfPics):
        minRange = i * chunkSize
        maxRange = min((i + 1) * chunkSize, wavenumbers.size - 1)
        if i == expectedNumbersOfPics - 1:
            maxRange = wavenumbers.size - 1
        printSinglePicMaximumAndMinimums(
            minRange,
            maxRange,
            realScatteredFields,
            absoluteScatteredFields,
            wavenumbers,
            frequencies,
        )


def printSinglePicMaximumAndMinimums(
    minRange,
    maxRange,
    realScatteredFields,
    absoluteScatteredFields,
    wavenumbers,
    frequencies,
):
    absoluteScatteredFields = absoluteScatteredFields[minRange:maxRange]
    absoluteMaximumIndex = np.argmax(absoluteScatteredFields, axis=0)
    wavenumbers = wavenumbers[minRange:maxRange]
    frequencies = frequencies[minRange:maxRange]
    for i in range(1, absoluteScatteredFields.size - 1):
        if (
            absoluteScatteredFields[i] > absoluteScatteredFields[i - 1]
            and absoluteScatteredFields[i] > absoluteScatteredFields[i + 1]
        ):
            absoluteMaximumIndex = i
    print(
        "\nAbsolute maximum: {} (Pa) --- Wavenumber: {} (1/m) --- Frequency: {} (Hz)".format(
            round(absoluteScatteredFields[absoluteMaximumIndex], 3),
            round(wavenumbers[absoluteMaximumIndex], 6),
            round(frequencies[absoluteMaximumIndex], 3),
        )
    )

    if realScatteredFields is None:
        return

    realScatteredFields = realScatteredFields[minRange:maxRange]
    realMaximumIndex = np.argmax(realScatteredFields, axis=0)
    realMinimumIndex = np.argmin(realScatteredFields, axis=0)
    for i in range(1, realScatteredFields.size - 1):
        if (
            realScatteredFields[i] > realScatteredFields[i - 1]
            and realScatteredFields[i] > realScatteredFields[i + 1]
        ):
            realMaximumIndex = i

        if (
            realScatteredFields[i] < realScatteredFields[i - 1]
            and realScatteredFields[i] < realScatteredFields[i + 1]
        ):
            realMinimumIndex = i

    print(
        "Real maximum:     {} (Pa) --- Wavenumber: {} (1/m) --- Frequency: {} (Hz)".format(
            round(realScatteredFields[realMaximumIndex], 2),
            round(wavenumbers[realMaximumIndex], 6),
            round(frequencies[realMaximumIndex], 3),
        )
    )
    print(
        "Real minimum:     {} (Pa) --- Wavenumber: {} (1/m) --- Frequency: {} (Hz)".format(
            round(realScatteredFields[realMinimumIndex], 2),
            round(wavenumbers[realMinimumIndex], 6),
            round(frequencies[realMinimumIndex], 3),
        )
    )


def plotSpatialField(maxValue, plotRange, absolute, real, imaginary, fieldType):
    minValue = -maxValue
    colorType = "bwr"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im = axes[0].imshow(
        absolute.T,
        cmap=colorType,
        vmin=minValue,
        vmax=maxValue,
        extent=[-plotRange, plotRange, -plotRange, plotRange],
    )
    axes[0].title.set_text("Absolute {} pressure".format(fieldType))
    axes[0].set_xlabel("z axis")
    axes[0].set_ylabel("y axis")

    axes[1].imshow(
        real.T,
        cmap=colorType,
        vmin=minValue,
        vmax=maxValue,
        extent=[-plotRange, plotRange, -plotRange, plotRange],
    )
    axes[1].title.set_text("Real {} pressure".format(fieldType))
    axes[1].set_xlabel("z axis")
    axes[1].set_ylabel("y axis")

    axes[2].imshow(
        imaginary.T,
        cmap=colorType,
        vmin=minValue,
        vmax=maxValue,
        extent=[-plotRange, plotRange, -plotRange, plotRange],
    )
    axes[2].title.set_text("Imaginary {} pressure".format(fieldType))
    axes[2].set_xlabel("z axis")
    axes[2].set_ylabel("y axis")

    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    #fig.suptitle("{} pressure (x=0)".format(fieldType), fontsize=16)
    plt.savefig('test.pdf', bbox_inches='tight')
    plt.show()
