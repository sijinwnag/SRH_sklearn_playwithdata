# %%-- import and setup
from lifetime_curve_plotting import *
from ipywidgets import interact
# %%-

# %%-- lifetime plot one curve
lifetime_obj = two_level_lifetime_plot()
lifetime_obj.plot_onecurve()
# %%-

# %%-- lifetime plot at differnet temperature
lifetime_obj = two_level_lifetime_plot(doping_type='p', doping=1.01e16, sigman1 = 1.55e-14, sigman2 = 1.53e-14, sigmap1 = 1.2e-14, sigmap2 = 1.39e-14, E1=0.159817862, E2=-0.225640063, Nt=1e12)
lifetime_obj.plot_swing_T(T_range=np.linspace(150, 400, 6))
# %%-

# %%-- lifetime plot at different doping level.
lifetime_obj = two_level_lifetime_plot(doping_type='p', doping=1.01e16, sigman1 = 1.55e-14, sigman2 = 1.53e-14, sigmap1 = 1.2e-14, sigmap2 = 1.39e-14, E1=0.159817862, E2=-0.225640063, Nt=1e12)
lifetime_obj.plot_swing_doping(doping_range=np.logspace(13, 19, 6))
# %%-

# %%-- lifetime plot at different Et1.
lifetime_obj = two_level_lifetime_plot()
lifetime_obj.plot_swing_Et1(Et_range=np.linspace(0.3, -0.3, 10))
# %%-

# %%-- interactive visualization
# see jupyter notebook
# %%-
