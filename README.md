# hybridastar_motion_planning

## requirement
- pytorch
- hybridastar service

## how to use
### 1.create datasets
~~~
python hybridastar_planning.py
~~~
Note: hybridastar service not included in this responsitory

### 2.train pnet
~~~
pnet_train.py
~~~

### 3.test pnet
~~~
neural_motion_planner.py
~~~
visualize: rviz

### todo
- [ ] increase network accuracy