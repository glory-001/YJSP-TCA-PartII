import math
import numpy as np
import matplotlib.pyplot as plt
from rocketcea.cea_obj import CEA_Obj
import pandas as pd

# getValue is a helper function for parsing through NASA CEA outputs from code snippets Kieran sent

def getValue(output, str, i=1, n=1): 
    # i is the col you're trying to get from the row of string (first item is 1)
    # n the "nth" appearance of this value in the output string that you want.
    ind = output.find(str) + len(str)
    # l is the length of the value we're finding
    l = 1
    if str == "Cp, KJ/(KG)(K)":
        n += 1
    # rho is formatted oddly in the output string, so it needs to be handled
    # differently. This can probably be made neater, but for now it works.
    if str == "RHO, KG/CU M":
        while i > 1:
            # When we're not yet at the point
            while output[ind] == " ":
                ind += 1
            # passing some earlier numbers
            while output[ind] != " ":
                ind += 1
            if output[ind:ind+2] == " 0":
                ind += 2
            i -= 1
        # figure out and return value if you're at its index:
        exp = 0
        while output[ind] == " ":
            ind += 1
        while output[ind + l] != " ":
            l += 1
            if output[ind + l] == "+":
                exp = int(output[ind + l + 1])
                break
            elif output[ind + l] ==  "-":
                exp = -1 * int(output[ind + l + 1])
                break
        return float(output[ind:ind+l]) * (10 ** exp)
    while ind >= 0 and n > 1:
        ind = output.find(str, ind + len(str)) + len(str)
        n -= 1
    # Some numbers have exponents, so that must be taken into account
    while i > 1:
        # When we're not yet at the point
        while output[ind] == " ":
            ind += 1
        # passing some earlier numbers
        while output[ind] != " ":
            ind += 1
        i -= 1
    # figure out and return value if you're at its index:
    while output[ind] == " ":
        ind += 1
    while output[ind + l] != " ":
        l += 1
    return float(output[ind:ind+l])

class Engine:
    def __init__(self, ox='LOX', fuel='RP1', cstar=.92, MR=2, numPts = 30, comb_psia=300): 
        self.ox = ox
        self.fuel = fuel
        self.cstar = cstar
        self.MR = MR
        self.numPts = 30
        self.comb_psia = comb_psia
        self.throatA = np.pi * 1.3900000000000001**2 # 1.39 is throat radius (currently hardcoded)
        self.conrat = 21.244549 / self.throatA

    def CEAoutput(self):
        ispObj = CEA_Obj(oxName=self.ox, fuelName=self.fuel, fac_CR=self.conrat)
        self.CEAreadout = ispObj.get_full_cea_output(self.comb_psia, self.MR, short_output=1, output="siunits") 
        self.CEAobj = ispObj

    def getEngineProps(self):
        data = np.genfromtxt('contour1.csv', delimiter=',', skip_header=1, usecols=(0, 1))
        data = data[~np.isnan(data).any(axis=1)]

        idxsChamber = np.linspace(0, 399, num=int(self.numPts/3), endpoint=True, dtype=int)
        idxsNozzle_c = np.linspace(400, 1596, num=int(self.numPts/3), endpoint=True, dtype=int)
        idxsNozzle_d = np.linspace(1651, 2094, num=int(self.numPts/3), endpoint=False, dtype=int) # !! why is there a huge jump here
        chamberProps = data[idxsChamber] # !! Z and R values for just the combustion chamber, cut this down to 30
        nozzleProps_c = data[idxsNozzle_c]
        nozzleProps_d = data[idxsNozzle_d]

        dataItems = ["Pinj/P ", "Ae/At", "MACH NUMBER", "CF", "Ivac, M/SEC", "Isp, M/SEC", "P, BAR", "T, K",
        "RHO, KG/CU M", "H, KJ/KG", "U, KJ/KG", "M, (1/n)", "Cp, KJ/(KG)(K)", "GAMMAs", "SON VEL,M/SEC",
        "VISC,MILLIPOISE", "CONDUCTIVITY  ", "PRANDTL NUMBER"] # 18

        # Items that do not have intector face data in the output string
        no_inj = ["Ae/At", "CF", "Ivac, M/SEC", "Isp, M/SEC"] # 4

        # Items that appear more than once
        repeatedItems = ["Cp, KJ/(KG)(K)", "CONDUCTIVITY  ", "PRANDTL NUMBER"] # 3
        for i in range(len(dataItems)):
            if dataItems[i] == "Ae/At":
                inj = getValue(self.CEAreadout, dataItems[i])
                end = getValue(self.CEAreadout, dataItems[i])
            elif dataItems[i] in no_inj:
                inj =  0
                end = getValue(self.CEAreadout, dataItems[i])
            else:
                inj = getValue(self.CEAreadout, dataItems[i])
                end = getValue(self.CEAreadout, dataItems[i], 2)
            chamberProps = np.column_stack((chamberProps, np.linspace(inj, end, int(self.numPts/3))))
        for i in range(len(repeatedItems)):
            # We want to ensure we're getting the frozen values
            inj = getValue(self.CEAreadout, repeatedItems[i], 1, 2)
            end = getValue(self.CEAreadout, repeatedItems[i], 2, 2)
            chamberProps = np.column_stack((chamberProps, np.linspace(inj, end, int(self.numPts/3)))) # !! parse through what this linspace does
        
        nozzleProps_c = np.column_stack((nozzleProps_c, np.zeros((len(nozzleProps_c), len(dataItems) + len(repeatedItems)))))
        nozzleProps_d = np.column_stack((nozzleProps_d, np.zeros((len(nozzleProps_d), len(dataItems) + len(repeatedItems)))))

        # generate converging nozzle properties
        for a in range(10):
            aRat = ((data[idxsNozzle_c[a], 1] ** 2) * math.pi)/ self.throatA 
            output = self.CEAobj.get_full_cea_output(self.comb_psia, self.MR, subar=aRat, short_output=1, output="siunits")
            
            # Grab properties
            for i in range(len(dataItems)):
                if dataItems[i] in no_inj:
                    prop = getValue(output, dataItems[i], 3)
                else:
                    prop = getValue(output, dataItems[i], 4)
                nozzleProps_c[a, i+2] = prop
            for i in range(len(repeatedItems)):
                prop = getValue(output, repeatedItems[i], 4, 2)
                nozzleProps_c[a, i + 2 + len(dataItems)] = prop

        # generate diverging nozzle properties
        for a in range(10):
            aRat = ((data[idxsNozzle_d[a], 1] ** 2) * math.pi )/ self.throatA
            output = self.CEAobj.get_full_cea_output(self.comb_psia, self.MR, eps=aRat, short_output=1, output="siunits")
            
            # Grab properties
            for i in range(len(dataItems)):
                if dataItems[i] in no_inj:
                    prop = getValue(output, dataItems[i], 3)
                else:
                    prop = getValue(output, dataItems[i], 4)
                nozzleProps_d[a, i+2] = prop
            for i in range(len(repeatedItems)):
                prop = getValue(output, repeatedItems[i], 4, 2)
                nozzleProps_d[a, i + 2 + len(dataItems)] = prop 

        engineProps = np.concatenate((chamberProps, nozzleProps_c, nozzleProps_d))
        engineProps[:, 0] = engineProps[:, 0] * 0.0254
        engineProps[:, 1] = engineProps[:, 1] * 0.0254

        self.engineProps = engineProps

    # exports values from CEA as xlsx / csv
    def export(self):
        df = pd.DataFrame(self.engineProps)
        df.to_excel("engineProps5.xlsx", index=False, header=False)

    def export_csv(self):
        df = pd.DataFrame(self.engineProps)
        df.to_csv("engineProps5.csv", index=False, header=False)