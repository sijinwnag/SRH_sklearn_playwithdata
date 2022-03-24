import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from semiconductor.recombination import SRH


class two_level_lifetime_plot():
    """
    This model only works for silicon
    """


    def __init__(self, T=300, doping_type='p', doping=1e15, sigman1 = 1e-15, sigman2 = 1e-15, sigmap1 = 1e-14, sigmap2 = 1e-14, E1=0.1, E2=-0.1, Nt=1e12):
        """
        1. define the basic default parameters
        """
        self.T = T
        self.doping = doping # unit is cm-3
        self.dopingtype = doping_type
        self.sigman1 = sigman1 # units are in cm2
        self.sigman2 = sigman2
        self.sigmap1 = sigmap1
        self.sigmap2 = sigmap2
        self.Et1 = E1 # unit is in eV and is compared to intrinsic fermi energy level
        self.Et2 = E2
        self.Nt = Nt # defect number density.


    # define the law of mass action
    def n_lawofmass(self, p, T):
        ni = 5.29e19*(T/300)**2.54*np.exp(-6726/T) # intrinsic carrier concentration of silicon at 300K cm-3
        n = ni**2/p0
        return n


    # the SRH densities
    def n1p1SRH(self, Et, T):
        # Nc: the effective density of states of silicon in conduction band
        Nv = 3.5e15*T**(3/2)
        Nc = 6.2e15*T**(3/2)
        # the bolzmans constant is always same
        k = 8.617e-5 # boltzmans coonstant, unit is eV/K
        n1 = Nc*np.exp(-(1.12-Et)/k/T)
        p1 = Nv*np.exp(-(Et-0)/k/T)
        return [n1, p1]


    # define the lifetime function for SRH recombination
    def SRHlifetime_two_level(self, dn, p0, n0, p1, n1, p2, n2, sigman1, sigman2, sigmap1, sigmap2, vn, vp, Nt):
        """
        This function aims to calculate the SRH lifetime of a defect of two energy level.
        input:
        dn: excess carrier concentration
        p0: hole concentration after doping
        n0: electron concentration after doping
        p1: the hole concentration if fermi energy is equal to defect level 1
        n1: the electron concentration if fermi energy is equal to defect level 1
        p2: the hole concentration if fermi energy is equal to defect level 2
        n2: the electron concentratoin if fermi energy is equal ot defect level 2
        sigman1: the electron capture cross sectional area for defect energy 1
        sigman2: the electron captrue cross sectional area for defect energy 2
        sigmap1: the hole capture cross sectional area for defect energy 1
        sigmap2: the hole capture cross sectional area for defect energy 2
        vn: thermal velocity of electron in the material (silicon)
        vp: thermal velocity of hole in the material (silicon)
        Nt: the number density of the defect
        output:
        tao: the SRH lifetime
        """
        # convert units into SI units
        # sigman1 = sigman1/1e4
        # sigman2 = sigman2/1e4
        # sigmap1 = sigmap1/1e4
        # sigmap2 = sigmap2/1e4
        # dn = dn*1e6
        # p0 = p0*1e6
        # n0 = n0*1e6
        # p1 = p1*1e6
        # p2 = p2*1e6
        # n1 = n1*1e6
        # n2 = n2*1e6
        # vn = vn/100
        # vp = vp/100
        # Nt = Nt * 1e6
        # calculate the total carrier concentration for n and p.
        n = n0 + dn
        p = p0 + dn
        # define the whole function into bracket and put them together
        bracket1 = (sigman1*vn*n1 + sigmap1*vp*p)/(sigmap1*vp*p1 + sigman1*vn*n)
        bracket2 = (sigmap2*vp*p2 + sigman2*vn*n)/(sigman2*vn*n2 + sigmap2*vp*p)
        bracket3 = n0 + p0 + dn
        bracket4 = (sigmap1*sigman1*vp*vn)/(sigmap1*vp*p1 + sigman1*vn*n)
        bracket5 = (sigman2*sigmap2*vn*vp)/(sigman2*vn*n2 + sigmap2*vp*p)
        # put together
        taoSRH = (1 + bracket1 + bracket2)/(Nt*bracket3*(bracket4 + bracket5))
        return taoSRH


    # defien the Auger recombination lifetime
    def Augerlifetime(self, n, dn):
        Cn = 2.8e-31
        Cp = 0.99e-31
        taoAuger = 1/(Cn*n**2 + Cp*n*dn)
        return taoAuger


    # define the radiative recombination
    def Radlifetime(self, p0, dn, T):
        # use law of mass action to calculate n0
        n0 = n_lawofmass(p0, T)
        # define B
        B = 9.5e-15
        radtao = 1/B/(n0 + p0 + dn)
        return radtao


    # define the SRH density using intrinsic carrier density
    def n1p1SRH2(self, Et_Ei, T, dn):
        # the Et in this equation is the Et-Ei instead of the Et-Ev
        vn, vp, ni = self.thermal_velocity(self.T, self.Nt, dn, self.doping, self.dopingtype) # intrinsic carrier
        # concentration of silicon addapted from PV education
        k = const.physical_constants['Boltzmann constant in eV/K'][0] # boltzmans coonstant, unit is eV/K
        # print(Et_Ei, T, dn, ni)
        n1 = ni*np.exp(Et_Ei/k/T)
        p1 = ni*np.exp(-Et_Ei/k/T)
        return n1, p1


    # define the capture time constant
    def taoptaon(self, sigmap, sigman, Nt, vp, vn):
        # calculate time constant
        taup = 1/sigmap/vp/Nt
        taun = 1/sigman/vn/Nt
        return [taup, taun]


    def DPSSSRHtau(self, Et_Ei, p0, n0, dn, k, taup0, T):
        # Et is the energy level of the defect relative to intrinsic fermi energy
        # p0 and n0 are the carrier densities at equilibrium after doping
        # dn is the excess carrier concentration
        # k is the ratio taun/taup
        # taup is the capture time constant for holes
        # calculate p1 and n1
        [n1, p1] = n1p1SRH2(Et_Ei, T)
        # calculate thermal velocity
        vp = np.sqrt(3*1.38e-23*T/(0.39*9.11e-31))
        vn = np.sqrt(3*1.38e-23*T/(0.26*9.11e-31))
        numerator = taup0*(p0 + p1 + dn)/k*(vp/vn) + taup0*(n0 + n1 + dn)
        denominator = p0 + n0 + dn
        return numerator/denominator


    def Green_1990(self, vals, temp, Egratio, **kargs):
        """
         This form as described by Green in 10.1063/1.345414.
         inputs:
            vals: (dic)
                the effect mass values
            temp: (float)
                the temperature in kelvin
        outputs:
            vel_th_c: (float)
                the termal velocity for the conduction in cm/s
            vel_th_v: (float)
                the termal velocity for the valance band in cm/s
        """

        # the values relative to the rest mass
        ml = vals['ml'] * const.m_e

        mt = vals['mt'] * Egratio * const.m_e

        delta = np.sqrt((ml - mt) / ml)
        # conduction band effective mass
        mth_c = 4. * ml / (
            1. + np.sqrt(ml / mt) * np.arcsin(delta) / delta)**2

        vel_th_c = np.sqrt(8 * const.k * temp / np.pi / mth_c)
        # valance band effective mass, its a 7 order poynomial fit
        mth_v = np.sum(
            [vals['meth_v' + str(i)] * temp**i for i in range(8)]) * const.m_e

        vel_th_v = np.sqrt(8 * const.k * temp / np.pi / mth_v)

        # adjust the values from m/s to cm/s and return
        return vel_th_c * 100, vel_th_v * 100


    def taop0ktaon0_liearDPSS(self, p1, n1, p0, slope, intercept, vn, vp):
        taop0 = ((1 + p1/p0)*slope + p1/p0*intercept)/(1-n1/p0 + p1/p0)
        taon0 = slope + intercept - taop0
        k = taop0*vp/taon0/vn
        return [taop0, taon0, k]


    def SRHlifetimegenerator(self, Et_Ei, vn, vp, dn, Nt, sigman, sigmap, T, p0):
        [n1, p1] = n1p1SRH2(Et_Ei, T)
        n0 = n_lawofmass(p0, T)
        [taop0, taon0] = taoptaon(sigmap, sigman, Nt, vp, vn)
        SRHtal = SRHlifetime(dn, p0, p1, n0, n1, taon0, taop0)
        return SRHtal


    def thermal_velocity(self, T, Nt, dn, doping, type='p'):
        """
        This function aims to calculate thermal velocity based on temperature T.
        """
        if type == "p": Tmodel=SRH(material="Si",temp = T, Nt = Nt, nxc = dn, Na = doping, Nd= 0, BGN_author = "Yan_2014fer")
        if type == "n": Tmodel=SRH(material="Si",temp = T, Nt = Nt, nxc = dn, Na = 0, Nd= doping, BGN_author = "Yan_2013fer")
        ni = Tmodel.nieff[0]
        Vn = Tmodel.vel_th_e[0]
        Vp = Tmodel.vel_th_h
        return Vn, Vp, ni


    def plot_onecurve(self, excess_range=np.logspace(12,17)):
        """
        excess range refers to the excess carrier concentration range that we want to plot on
        """
        # we will swing across different doping level
        # create an empty list for tau:
        taulist = []
        for dn in excess_range:
            # calculate the thermal thermal velocity.
            vn, vp, ni = self.thermal_velocity(self.T, self.Nt, dn, self.doping, self.dopingtype)
            # calculate the minority carrier concentration:
            # Calculate n0 and p0
            if self.dopingtype == "p":
                p0 = (0.5 * (np.abs(self.doping - 0) + np.sqrt((0 - self.doping)**2 + 4 * ni**2)))
                n0 = (ni**2)/p0
            if self.dopingtype == "n":
                n0 = (0.5 * (np.abs(self.doping - 0) + np.sqrt((0 - self.doping)**2 + 4 * ni**2)))
                p0 = (ni**2)/n0
            # calculate n1 and p1 and n2 and p2
            n1, p1 = self.n1p1SRH2(self.Et1, self.T, dn)
            n2, p2 = self.n1p1SRH2(self.Et2, self.T, dn)
            # calculate the lifetime: self, dn, p0, n0, p1, n1, p2, n2, sigman1, sigman2, sigmap1, sigmap2, vn, vp, Nt
            tau = self.SRHlifetime_two_level(dn=dn, p0=p0, n0=n0, p1=p1, n1=n1, p2=p2, n2=n2, sigman1=self.sigman1, sigman2=self.sigman2, sigmap1=self.sigmap1, sigmap2=self.sigmap2, vn=vn, vp=vp, Nt=self.Nt)
            taulist.append(tau)

        # plot tau as a function of excess carrier concentration
        plt.plot(excess_range, taulist)
        plt.xscale('log')
        plt.yscale('log')


    def plot_swing_T(self, excess_range=np.logspace(12,17), T_range=np.linspace(290, 310, 3)):
        """
        This plot aims to swing T for lifetime curve.
        """
        plt.figure()
        plt.title('two-level defect lifeitme at different temperature')
        plt.xlabel('excess carrier concentration $cm^{-3}$')
        plt.ylabel('lifetime (s)')
        # we will swing across different temperature
        labellist = []
        for T in T_range:
            self.T = T
            self.plot_onecurve(excess_range=excess_range)
            labellist.append(str(int(round(T))) + 'K')
        plt.legend(labellist)
        plt.show()


    def plot_swing_doping(self, excess_range=np.logspace(12,17), doping_range=np.logspace(13, 15, 3)):
        """
        This plot aims to swing doping for lifetime curve.
        """
        plt.figure()
        plt.title('two-level defect lifeitme at different doping level')
        plt.xlabel('excess carrier concentration $cm^{-3}$')
        plt.ylabel('lifetime (s)')
        # we will swing across different doping.
        labellist = []
        for doping in doping_range:
            self.doping = doping
            self.plot_onecurve(excess_range=excess_range)
            labellist.append(str(np.format_float_scientific(doping, precision = 1, exp_digits=3)) + '$cm^{-3}$')
        plt.legend(labellist)
        plt.show()


    def plot_swing_Et1(self, excess_range=np.logspace(12,17), Et_range=np.linspace(0.3, -0.3, 10)):
        """
        This plot aims to swing doping for lifetime curve.
        """
        plt.figure()
        plt.title('two-level defect lifeitme at different $E_{t1}$')
        plt.xlabel('excess carrier concentration $cm^{-3}$')
        plt.ylabel('lifetime (s)')
        # we will swing across different temperature
        labellist = []
        for Et1 in Et_range:
            self.Et1 = Et1
            self.plot_onecurve(excess_range=excess_range)
            labellist.append(str(round(Et1, 3)) + 'eV')
        plt.legend(labellist)
        plt.show()
