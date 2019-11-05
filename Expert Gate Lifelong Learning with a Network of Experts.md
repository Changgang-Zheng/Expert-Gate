## Expert Gate: Lifelong Learning with a Network of Experts

#### How to learn such a gate function to differentiate between tasks, without having access to the training data of previous tasks


 

 
<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-02-18 at 5.13.56 PM.png"  alt="Transfer Learning" width="500"/></center>

### Selecting the most relevant expert

The reconstruction error eri of the i-th autoencoder is the output of the loss function given the input sample x.

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-02-18 at 5.14.19 PM.png"  alt="Transfer Learning" width="400"/></center>



###  Measuring task relatedness

Since we do not have access to the data of task Ta, we use the validation data from the current task Tk. We compute the average reconstruction error Erk on the current task data made by the current task autoencoder Ak and, likewise, the average reconstruction error Era made by the previous task autoencoder Aa on the current task data.

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-02-18 at 5.14.26 PM.png"  alt="Transfer Learning" width="400"/></center>

### We exploit task relatedness in two ways. 
* First, we use it to select the most related task to be used as prior model for learning the new task. 
* Second, we exploit the level of task relatedness to determine which transfer method to use: fine- tuning or learning-without-forgetting (LwF) 