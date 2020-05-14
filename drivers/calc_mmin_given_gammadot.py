from astropy import units as u

tau = 2.66*u.year
gammadot = 50*u.m/u.s/u.day
mstar = 0.35 * u.Msun

m_min = 6*u.Mjup * (tau/(1*u.year))**(4/3) * (gammadot/(1*u.m/u.s/u.day)) * (mstar/(1*u.Msun))**(2/3)

print(m_min.to(u.Msun))
