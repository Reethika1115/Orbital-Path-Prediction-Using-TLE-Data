# Orbital-Path-Prediction-Using-TLE-Data

## TLE(Two-Line Element )

The orbital parameters of a satellite can be summarized into two lines of information with a maximum of 69 alphanumeric characters, which is why this is called “Two-Line Element (TLE)”

![Screenshot 2024-12-29 134454](https://github.com/user-attachments/assets/81352af9-9b96-4921-9392-7f332f76a052)

## SGP4 (Simplified General Perturbations) & DSGP4(Differentiable Simplified General Perturbations)
I used dspg4 because of its

Integration with pytorch for better tensor operations

Batch processing ability and GPU acceleration

Machine learning Extension

SGP4 indicates the effects of atmospheric drag and the effects of the non-uniform gravity due to the earth’s orbit shape and influence of moon and sun.
There are 3 types of rotations to be done to extract positions of the satellite at different timestamps.
## Rotate along the Z-axis(RAAN)

Right Ascension of the Ascending Node (RAAN) determines the orientation of the ascending node where the satellite crosses the equatorial plane.This rotation aligns the satellite's orbital plane with the celestial coordinate system
## Rotate along the New X-axis(Inclination)

Inclination tilts the satellite's orbital plane relative to the Earth's equatorial plane.This rotation adjusts the satellite's path to match the specified orbital inclination.
## Rotate within the orbital plane(Argument of Perigee)

Argument of Perigee orients the elliptical orbit within its plane, indicating the location of the closest approach to Earth.This rotation defines the position of the perigee within the orbital plane.

dSGP4/SPG4 automate the application of these rotations as part of their core functionalities. We don't need to manually compute and apply each rotation; the algorithms handle these transformations internally.

### tsinces = torch.cat([torch.linspace(0,2*60,20000)]*len(tles))

This line divides 120 min time into 20000 timestamps for each TLE 

### _,tle_batch=dsgp4.initialize_tle(tles_)                                                                                                                                            
This line initialize the tles for the dspg4 propagation

### states_teme=dsgp4.propagate_batch(tle_batch,tsinces)

This line propagates the batch and gives the output of positions and velocities stored in states_teme.
Since the list of tles has four tle data, the tles of each 20000 timestamps appended in order, 2 indicted positions and velocities and 3 indicates their three components(x,y,z)

The output tensor is in TEME frame(True Equator Mean Equinox)

https://drive.google.com/drive/folders/1dEXrlkfbxqAnTFAA1CNetl68E-RymIn6?usp=drive_link

For further analysis of the position I tried to project the orbital path of ISS on a world map.To do this first the TEME frame has to be changed to ECEF (Earth-centered Earth-Fixed) coordinates and then to geodetic coordinates.
### TEME FRAME

-TEME is an inertial coordinate system used for propagating satellite orbits. It is not fixed to the Earth's surface but is aligned with the true equator and mean equinox.
### ECEF FRAME

-ECEF is a coordinate system fixed to the Earth, rotating with it. It provides a stable frame to express positions relative to the Earth’s surface.
WGS84 Model for converting ECEF coordinates to Geodetic coordinates (Longitude,Latitude,Altitude) these are the constants used in this model.

a = 6378.137  # Semi-major axis

f = 1 / 298.257223563  # Flattening

e_squared = 2 * f - f**2  # Eccentricity squared

Since Earth has the ellipsoid shape Latitude is calculated in iterative process with initial guess of earth  spherical shape .


