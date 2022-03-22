# %%-- import and setup
from lifetime_curve_plotting import *

lifetime_obj = two_level_lifetime_plot()
# %%-

# %%-- lifetime plot one curve
lifetime_obj.plot_onecurve()
# %%-

# %%-- lifetime plot at differnet temperature
lifetime_obj.plot_swing_T()
# %%-

# %%-- lifetime plot at different doping level.
lifetime_obj.plot_swing_doping()
# %%-
