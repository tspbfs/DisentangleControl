

# Code: Disentangling Linear Quadratic Control with Untrusted ML Predictions


## Setup

Environment setup
```python
conda create --name ENV python=3.11
pip install -r requirements.txt
```


Tasks are organized as follows:
```
├── src
│   ├── tracking
│   ├── voltage
```

## Running
### Drone Tracking 
```python
cd src/tracking
./_run_tracking.sh
```
### Voltage Control
```python
cd src/tracking
./_run_voltage.sh
```