# simcosinor
[![Build Status](https://travis-ci.org/trislett/simcosinor.svg?branch=master)](https://travis-ci.org/trislett/simcosinor)

Simulations for the cosinor model.

#### Lazy install
```
virtualenv -p python3.7 python37env
source python37env/bin/activate
git clone https://github.com/trislett/simcosinor.git
cd simcosinor
pip install .
cd ..
```

#### Some examples

Just run 10000 simulations using the defaults: 
```
simcosinor -e threesubs_modality1
```

Run the simulations using 24 random times points only.

```
simcosinor -e threesubs_modality1 -rand -ns 24
```

Run the simulations with eveningly dispursed time-points, but limited from 11:30am until 11:30pm.

```
simcosinor -e threesubs_modality1 -rand -ns 72 -sr 11.5 23.5 -er
```

Run the simulations with randomly dispursed time-points.

```
simcosinor -e threesubs_modality1 -rand -ns 72 -sr 0 24
```

### Plotting examples

Run simulation and generate plots of the right insula gyrus

```
simcosinor -e threesubs_modality1 -rand -ps -pp -pw 24 -ppm -roi rh.R_Ig
```

Console output:

```
Running 10000 simulations...
ROI = rh.R_Ig
[Metric]		[Mean] [Standard Deviation]
R2		=	0.3062 [0.0832]
Acro24[24.0]	=	16.8039 [0.7275]
-logP		=	12.8674 [4.2322]
```

Simulation plot:

![Simulation Plot](simcosinor/examples/R_Ig_cosinor_simulation_plot.png)

Residual plot of the cosinor model (used to estimate the error):

![Simulation Plot Residuals](simcosinor/examples/R_Ig_cosinor_simulation_plot_residuals.png)

Periodogram:

![Periodogram](simcosinor/examples/R_Ig_periodogram_plot.png)


Sliding window (size = 24):

![Sliding window](simcosinor/examples/R_Ig_sliding_window_plot.png)

Cosinor plot with permutation testing:

![Permutations](simcosinor/examples/R_Ig_cosinor_plot_permuted.png)

