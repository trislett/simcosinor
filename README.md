# simcosinor
Simulations for the cosinor model.

#### Lazy install
```
virtualenv -p python3.7 python37env
source python37env/bin/activate
git clone https://github.com/trislett/simcosinor.git
cd simcosinor
pip install .
```

#### Some examples

Just run 10000 simulations using the defaults: 
```
simcosinor
```

Run the simulations using 24 random times points only.

```
simcosinor -rand -ns 24
```

Run the simulations with evenling dispursed time-points, but limited from 11:30am until 11:30pm.

```
simcosinor -rand -ns 72 -sr 11.5 23.5 -er
```

Run the simulations with randomly dispursed time-points.

```
simcosinor -rand -ns 72 -sr 0 24
```
