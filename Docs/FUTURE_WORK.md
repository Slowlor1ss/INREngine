# Future Work

## **Planned Features**
* [ ] **Automatic INR Configuration Optimizer**:
  * Implement Hyperparameter Optimization (HPO) algorithms like BOHB and ASHA. The goal is to build an automated system that looks for the best hyperparameters to run the network with for a given data set or possibly in general? Still doing research on this, so not much to write about yet.
* [ ] **MRI intermidiate slice generator**:
  * One geniune use for INR's which has already been implemented in practice; is the generation of in between slices of a MRI scan, INR's are perfect for this as they dont hallucinate unline generative models as INR's have no knoledge of what they are making, INR are basically just a mathmatical representation of the training data and virtually scale infinitly, which makes them perfect for thsi use case. 
* [ ] **NeRFs**:
  * Extend spatial derivatives (delta x, delta y, delta z) and grid encoders to support NeRFs and 3D Signed Distance Functions (SDFs).
  [Nvidea talk](https://tom94.net/data/publications/mueller22instant/mueller22instant-rtl.mp4)