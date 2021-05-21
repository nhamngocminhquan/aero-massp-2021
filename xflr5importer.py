import aerosandbox as asb
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import os


class AirfoilPolars:
    def __init__(self, filePrefix="", fileDirectory="", CLfitOrder=3, CDpfitOrder=2, CmfitOrder=3,
                 weightedFit=False, plotFit=False):
        """
        Class containing airfoil read from file, as well as CL, CDp and Cm interpolation functions

        Point pairs (Re, AOA) as well as CL, CDp and Cm at each point is .points and .values(CL/CDp/Cm)

        :param filePrefix: an identifier for the airfoil file names, such as "NACA 0012_"
        :param fileDirectory: location of files
        :param fitOrder: degree of polynomial fit for CL, CDp and Cm, default is 3 for performance
        :param weightedFit: if True, the reader will add weight to polars near 2.5deg AOA
                            using a cos function with period T = 40deg
        :param plotFit: if True, the data will be scatter-plotted with poly fits
        """

        # Airfoil instance with characteristics read from files
        self.airfoil = None

        # Polars read from file
        self.polars = []

        # Point pairs (Re, AOA) as well as CL CDp and Cm at each point
        self.points = []

        self.valuesCL = []
        self.valuesCDp = []
        self.valuesCm = []

        # A list of Reynolds numbers
        self.Res = []

        # AOA, CL, CDp, Cm at each Re
        self.AOAs_Re = []
        self.CLs_Re = []
        self.CDps_Re = []
        self.Cms_Re = []

        # Fit characteristics
        self.CLfitOrder = CLfitOrder
        self.CDpfitOrder = CDpfitOrder
        self.CmfitOrder = CmfitOrder
        self.weightedFit = weightedFit
        self.plotFit = plotFit

        # Polyfit coeffs at each Re w.r.t. angle
        self.CLfit_Re = []
        self.CDpfit_Re = []
        self.Cmfit_Re = []

        # Keep some handy flags
        self.importedPolars = False
        self.createdCPolyfitTables = False

        if filePrefix is not "" and fileDirectory is not "":
            print("\nReading from xflr5 files...")
            try:
                # Try getting polars from file
                self.xflr5AirfoilPolarReader(filePrefix, fileDirectory)
                self.importedPolars = True

            except:
                self.importedPolars = False
                print("Read unsuccessful!")

            if self.importedPolars:
                print("Read successful!")
                print("\nCreating polynomial fits for coefficients...")
                try:
                    # Create lookup tables for CL, CDp and Cm
                    self.CreateCoefficientPolyfitTables()
                    self.createdCPolyfitTables = True

                except:
                    print("Fit unsuccessful!")
                    self.createdCPolyfitTables = False

            if self.createdCPolyfitTables:
                print("Fit successful!")

                # Create Airfoil instance
                self.airfoil = asb.Airfoil(
                    CL_function=lambda alpha, Re, mach, deflection: (  # Lift coefficient function
                        self.CL_function(Re * 10e-6, alpha)
                    ),
                    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
                        self.CDp_function(Re * 10e-6, alpha)
                    ),
                    Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
                        self.Cm_function(Re * 10e-6, alpha)
                    )
                )

                try:
                    self.PlotPolyFit()

                except:
                    print("Plot unsuccessful")

    def xflr5AirfoilPolarReader(self, filePrefix, fileDirectory):
        """
        Read polars from xflr5 output.

        :param filePrefix: Name/prefix/identifier for the airfoil
        :param fileDirectory: Directory containing files
        :return: Dict of
        """
        for filename in os.listdir(fileDirectory):
            if filePrefix in filename:
                try:
                    # Get Re number (times 10^6)
                    currentRe = float(filename[filename.find("Re") + 2:filename.find("Re") + 7])

                    # Open and detect "----" line
                    fp = open(fileDirectory + "/" + filename)

                    # Keep a value detection flag
                    valuesFound = False

                    # Keep a Re-specific output polar list
                    currentPolarsOutput = []

                    for line in fp:
                        if "----" in line:
                            valuesFound = True
                            continue

                        if valuesFound and len(line) > 2:
                            # Get AOA, CL, CD, CDp and Cm, then pop CD
                            # Then convert to float and attach to currentPolarsOutput
                            polar = line.strip().split()[:5]
                            polar.pop(2)

                            currentPolarsOutput.append([float(i) for i in polar])

                    currentPolarsOutput = sorted(currentPolarsOutput, key=lambda l: l[0])
                    self.polars.append([currentRe, currentPolarsOutput])
                    self.Res.append(currentRe)

                except:
                    print("An exception occurred")

                finally:
                    fp.close()

        self.polars = sorted(self.polars, key=lambda l: l[0])
        self.Res = sorted(self.Res)

    def CreateCoefficientPolyfitTables(self):
        """
        Create organized polars for further use

        A pair of Reynolds number and angle of attack corresponding to CL, CDp and Cm values
        """
        for currentRe in self.polars:
            for currentPolar in currentRe[1]:
                # Combine (Re, AOA) as a point
                self.points.append([currentRe[0], currentPolar[0]])

                # Corresponding to CL, CDp and Cm value
                self.valuesCL.append(currentPolar[1])
                self.valuesCDp.append(currentPolar[2])
                self.valuesCm.append(currentPolar[3])

        for Re in self.Res:
            self.AOAs_Re.append([point[1] for point in self.points if point[0] == Re])
            self.CLs_Re.append([self.valuesCL[i] for i in range(len(self.points)) if self.points[i][0] == Re])
            self.CDps_Re.append([self.valuesCDp[i] for i in range(len(self.points)) if self.points[i][0] == Re])
            self.Cms_Re.append([self.valuesCm[i] for i in range(len(self.points)) if self.points[i][0] == Re])

            # Fit coefficients to AOAs
            if self.weightedFit:
                w = np.array([np.cos((a - 2.5) / 5 * np.pi / 4) for a in self.AOAs_Re[-1]])
                # Fit coefficients with weight
                self.CLfit_Re.append(np.polyfit(self.AOAs_Re[-1], self.CLs_Re[-1],
                                                self.CLfitOrder, w=w))
                self.CDpfit_Re.append(np.polyfit(self.AOAs_Re[-1], self.CDps_Re[-1],
                                                 self.CDpfitOrder, w=w))
                self.Cmfit_Re.append(np.polyfit(self.AOAs_Re[-1], self.Cms_Re[-1],
                                                self.CmfitOrder, w=w))

            else:
                self.CLfit_Re.append(np.polyfit(self.AOAs_Re[-1], self.CLs_Re[-1], self.CLfitOrder))
                self.CDpfit_Re.append(np.polyfit(self.AOAs_Re[-1], self.CDps_Re[-1], self.CDpfitOrder))
                self.Cmfit_Re.append(np.polyfit(self.AOAs_Re[-1], self.Cms_Re[-1], self.CmfitOrder))

    def PlotPolyFit(self):
        """
        Plot coefficient fits
        """
        if self.plotFit:
            # Plot fit output
            xp = np.linspace(-10, 10, 30)
            fig = plt.figure()
            axCL = fig.add_subplot(1, 3, 1, projection='3d')
            axCDp = fig.add_subplot(1, 3, 2, projection='3d')
            axCm = fig.add_subplot(1, 3, 3, projection='3d')
            for i in range(len(self.Res)):
                colorCoefx = i * 1.0 / len(self.Res)
                colorCoef = colorCoefx ** 2
                axCL.scatter3D(np.ones(len(self.AOAs_Re[i])) * np.log(self.Res[i]),
                               self.AOAs_Re[i], self.CLs_Re[i], alpha=0.5, s=1.5,
                               color=(colorCoef ** 2, 1 - colorCoef, - 4 * (colorCoef - 0.5) ** 2 + 1))

                axCL.plot(np.ones(len(xp)) * np.log(self.Res[i]),
                          xp, self.PolyEval(xp, self.CLfit_Re[i]), lw=0.6,
                          color=(colorCoef ** 2, 1 - colorCoef, - 4 * (colorCoef - 0.5) ** 2 + 1))

                axCDp.scatter3D(np.ones(len(self.AOAs_Re[i])) * np.log(self.Res[i]),
                                self.AOAs_Re[i], self.CDps_Re[i], alpha=0.5, s=1.5,
                                color=(colorCoef ** 2, 1 - colorCoef, - 4 * (colorCoef - 0.5) ** 2 + 1))

                axCDp.plot(np.ones(len(xp)) * np.log(self.Res[i]),
                           xp, self.PolyEval(xp, self.CDpfit_Re[i]), lw=0.6,
                           color=(colorCoef ** 2, 1 - colorCoef, - 4 * (colorCoef - 0.5) ** 2 + 1))

                axCm.scatter3D(np.ones(len(self.AOAs_Re[i])) * np.log(self.Res[i]),
                               self.AOAs_Re[i], self.Cms_Re[i], alpha=0.5, s=1.5,
                               color=(colorCoef ** 2, 1 - colorCoef, - 4 * (colorCoef - 0.5) ** 2 + 1))

                axCm.plot(np.ones(len(xp)) * np.log(self.Res[i]),
                          xp, self.PolyEval(xp, self.Cmfit_Re[i]), lw=0.6,
                          color=(colorCoef ** 2, 1 - colorCoef, - 4 * (colorCoef - 0.5) ** 2 + 1))

            # Set axes labels
            axCL.set(xlabel="ln(Re / 10^6)", ylabel="Angle of attack (degrees)", zlabel="CL")
            axCDp.set(xlabel="ln(Re / 10^6)", ylabel="Angle of attack (degrees)", zlabel="CDp")
            axCm.set(xlabel="ln(Re / 10^6)", ylabel="Angle of attack (degrees)", zlabel="Cm")

            plt.show(block=True)

    def PolyEval(self, x, coefs):
        returnVal = coefs[0]
        for i in range(1, len(coefs)):
            returnVal = returnVal * x + coefs[i]

        return returnVal

    def CL_function(self, Re, angle):
        # 2 ref values for Re
        refRe = [0, 0]
        refCL = [0, 0]

        for i in range(1, len(self.Res)):
            refRe = if_else(self.Res[i] > Re,
                            if_else(Re > self.Res[i - 1], [self.Res[i - 1], self.Res[i]], refRe),
                            refRe)

        refRe = if_else(self.Res[0] > Re, [self.Res[0], self.Res[1]],
                        if_else(self.Res[-1] < Re, [self.Res[-2], self.Res[-1]], refRe))

        for i in range(1, len(self.Res)):
            refCL[0] = if_else(self.Res[i] == refRe[1],
                               self.PolyEval(angle, self.CLfit_Re[i - 1]),
                               refCL[0])

            refCL[1] = if_else(self.Res[i] == refRe[1],
                               self.PolyEval(angle, self.CLfit_Re[i]),
                               refCL[1])

        # If an exception is thrown, the operating point extends beyond Reynolds number range
        tester = 1 / (refRe[1] - refRe[0])
        return (Re - refRe[0]) / (refRe[1] - refRe[0]) * (refCL[1] - refCL[0]) + refCL[0]

    def CDp_function(self, Re, angle):
        # 2 ref values for Re
        refRe = [0, 0]
        refCDp = [0, 0]

        for i in range(1, len(self.Res)):
            refRe = if_else(self.Res[i] > Re,
                            if_else(Re > self.Res[i - 1], [self.Res[i - 1], self.Res[i]], refRe),
                            refRe)

        refRe = if_else(self.Res[0] > Re, [self.Res[0], self.Res[1]],
                        if_else(self.Res[-1] < Re, [self.Res[-2], self.Res[-1]], refRe))

        for i in range(1, len(self.Res)):
            refCDp[0] = if_else(self.Res[i] == refRe[1],
                                self.PolyEval(angle, self.CDpfit_Re[i - 1]),
                                refCDp[0])

            refCDp[1] = if_else(self.Res[i] == refRe[1],
                                self.PolyEval(angle, self.CDpfit_Re[i]),
                                refCDp[1])

        # If an exception is thrown, the operating point extends beyond Reynolds number range
        tester = 1 / (refRe[1] - refRe[0])
        return (Re - refRe[0]) / (refRe[1] - refRe[0]) * (refCDp[1] - refCDp[0]) + refCDp[0]

    def Cm_function(self, Re, angle):
        # 2 ref values for Re
        refRe = [0, 0]
        refCm = [0, 0]

        for i in range(1, len(self.Res)):
            refRe = if_else(self.Res[i] > Re,
                            if_else(Re > self.Res[i - 1], [self.Res[i - 1], self.Res[i]], refRe),
                            refRe)

        refRe = if_else(self.Res[0] > Re, [self.Res[0], self.Res[1]],
                        if_else(self.Res[-1] < Re, [self.Res[-2], self.Res[-1]], refRe))

        for i in range(1, len(self.Res)):
            refCm[0] = if_else(self.Res[i] == refRe[1],
                               self.PolyEval(angle, self.Cmfit_Re[i - 1]),
                               refCm[0])

            refCm[1] = if_else(self.Res[i] == refRe[1],
                               self.PolyEval(angle, self.Cmfit_Re[i]),
                               refCm[1])

        # If an exception is thrown, the operating point extends beyond Reynolds number range
        tester = 1 / (refRe[1] - refRe[0])
        return (Re - refRe[0]) / (refRe[1] - refRe[0]) * (refCm[1] - refCm[0]) + refCm[0]

    def KindOfBilinearApproximation(self, point, refPoints, refValues):
        """
        Input a set of 4 2-D reference points and corresponding reference values, return value at 2-D point
        https://math.stackexchange.com/a/832635

        :param point:
        :param refPoints:
        :param refValues:
        :return: doesnt work
        """
        kind = 'inner'
        tempX = []
        tempZ = []
        for i in range(len(refPoints)):
            x = refPoints[i][0]
            y = refPoints[i][1]
            tempX.append([x ** 2, x * y, y ** 2, x, y, 1])
            tempZ.append(refValues[i])

        X = SX(tempX)
        Z = vertcat(SX.zeros(6, 1), SX(tempZ))
        E = vertcat(horzcat(SX.eye(3) + SX.zeros(3, 3), SX.zeros(3, 3)), SX.zeros(3, 6))
        A = vertcat(horzcat(E, X.T), horzcat(X, SX.zeros(4, 4)))

        a = solve(A, Z)
        # print(X)
        # print(Z)
        # print(a)
        x = point[0]
        y = point[1]
        return a[0] * (x ** 2) + a[1] * (x * y) + a[2] * (y ** 2) + a[3] * x + a[4] * y + a[5]


"""
Code bank for future revision

                # Create interpolate functions for CL, CDp and Cm
                # tempPoints = np.asarray(self.points)
                tempReCL = []
                currentRe = self.points[0][0]
                for i in range(len(self.points)):
                    if self.points[i][0] > currentRe:
                        self.CL_Re_interpolators.append(currentRe, interp1d(np.asarray(tempReCL)[:, 0], np.asarray(tempReCL)[:, 1]))

                        tempReCL = []
                        currentRe = self.points[i][0]

                    tempReCL.append([self.points[i][1], self.valuesCL[i]])

                #self.CL_Re_interpolators.append()
                #self.CL_function = ip.interp2d(tempPoints[:, 0], tempPoints[:, 1], self.valuesCL)


        # ip.interp2d()
       CL_interpolator = None
        Re_CL = []
        for interpolator in self.CL_Re_interpolators:
            print(interpolator[1](angle))
            Re_CL.append([interpolator[0], interpolator[1](angle)])
            print([interpolator[0], interpolator[1](angle)])

        tempRe_CL = np.asarray(Re_CL)
        CL_interpolator = interp1d(tempRe_CL[:, 0], tempRe_CL[:, 1])

        # This should be done beforehand
        tempAOA = [[point[1] for point in self.points if point[0] == refRe[0]],
                   [point[1] for point in self.points if point[0] == refRe[1]]]

        tempVal = [[self.valuesCL[i] for i in range(len(self.points)) if self.points[i][0] == refRe[0]],
                   [self.valuesCL[i] for i in range(len(self.points)) if self.points[i][0] == refRe[1]]]

        indexes = [0, 0]

        for i in range(len(self.Res)):
            indexes[0] = if_else(refRe[0] == self.Res[i], i, indexes[0])
            indexes[1] = if_else(refRe[1] == self.Res[i], i, indexes[1])

        tempAOA = [self.AOAs_Re[indexes[0].getValue()], self.AOAs_Re[indexes[1].getValue()]]
        tempVal = [self.CLs_Re[indexes[0].getValue()], self.CLs_Re[indexes[1].getValue()]]


        for i in range(2):
            refPoints.append([0, 0])

            for j in range(1, len(tempAOA[i])):
                refPoints[-1] = if_else(tempAOA[i][j] > angle > tempAOA[i][j - 1], [refRe[i], tempAOA[i][j - 1]], refPoints[-1])

            refPoints[-1] = if_else(angle < tempAOA[i][0], [refRe[i], tempAOA[i][0]],
                                    if_else(angle > tempAOA[i][-1], [refRe[i], tempAOA[i][-2]], refPoints[-1]))

            refPoints.append([0, 0])
            for j in range(1, len(tempAOA[i])):
                refPoints[-1] = if_else(tempAOA[i][j] > angle > tempAOA[i][j - 1], [refRe[i], tempAOA[i][j]],
                                        refPoints[-1])

            refPoints[-1] = if_else(angle < tempAOA[i][0], [refRe[i], tempAOA[i][1]],
                                    if_else(angle > tempAOA[i][-1], [refRe[i], tempAOA[i][-1]], refPoints[-1]))

            refValues.append([0, 0])
            for j in range(1, len(tempAOA[i])):
                refValues[-1] = if_else(tempAOA[i][j] > angle > tempAOA[i][j - 1], tempVal[i][j - 1],
                                        refValues[-1])

            refValues[-1] = if_else(angle < tempAOA[i][0], tempVal[i][0],
                                    if_else(angle > tempAOA[i][-1], tempVal[i][-2], refValues[-1]))

            refValues.append([0, 0])
            for j in range(1, len(tempAOA[i])):
                refValues[-1] = if_else(tempAOA[i][j] > angle > tempAOA[i][j - 1], tempVal[i][j],
                                        refValues[-1])

            refValues[-1] = if_else(angle < tempAOA[i][0], tempVal[i][1],
                                    if_else(angle > tempAOA[i][-1], tempVal[i][-1], refValues[-1]))

        if Re < self.Res[0]:
            refRe = [self.Res[0], self.Res[1]]

        elif Re > self.Res[-1]:
            refRe = [self.Res[-2], self.Res[-1]]

        else:
            for i in range(1, len(self.Res)):
                if self.Res[i] > Re > self.Res[i - 1]:
                    refRe = [self.Res[i - 1], self.Res[i]]
                    pass

        # This should be done beforehand
        tempAOA = [[point[1] for point in self.points if point[0] == refRe[0]],
                   [point[1] for point in self.points if point[0] == refRe[1]]]

        tempVal = [[self.valuesCL[i] for i in range(len(self.points)) if self.points[i][0] == refRe[0]],
                   [self.valuesCL[i] for i in range(len(self.points)) if self.points[i][0] == refRe[1]]]

        for i in range(2):
            if angle < tempAOA[i][0]:
                refPoints.append([refRe[i], tempAOA[i][0]])
                refValues.append(tempVal[i][0])
                refPoints.append([refRe[i], tempAOA[i][1]])
                refValues.append(tempVal[i][1])

            elif angle > tempAOA[i][-1]:
                refPoints.append([refRe[i], tempAOA[i][-2]])
                refValues.append(tempVal[i][-2])
                refPoints.append([refRe[i], tempAOA[i][-1]])
                refValues.append(tempVal[i][-1])

            else:
                for j in range(1, len(tempAOA[i])):
                    if tempAOA[i][j] > angle > tempAOA[i][j - 1]:
                        refPoints.append([refRe[i], tempAOA[i][j - 1]])
                        refValues.append(tempVal[i][j - 1])
                        refPoints.append([refRe[i], tempAOA[i][j]])
                        refValues.append(tempVal[i][j])
                        pass


"""
