# Network_optimization
The proposed KG-driven and DDQN network optimization approaches

Create a NetworkOpt virtual environment
```conda create -n NetworkOpt python=3.12```

Activate the virtual environment
```conda activate NetworkOpt```

Run this to install the required libraries:  
```pip install -r requirements.txt```  

Run this code:
 ```cd optimizeAlgorithm```

KG-driven:       

```python Dueling_DDQN_cover1.py```

Train model: Set ```is_train=True``` in line 293

Test model: Set ```is_train=False``` in line 293

DDQN:        

```python Dueling_DDQN_cover_total.py```

Train model: Set ```is_train=True``` in line 355

Test model: Set ```is_train=False``` in line 355

Show figure:

```python show_figure.py```