---
permalink: /
title: "Machine learning enable fusion of CAE data and test data for vehicle crashworthiness performance evalaution by analysis"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
<style>
p {
    text-align: justify;
}
</style>

Introduction: The Intersection of Safety and Innovation
======
A crucial component of automotive engineering is vehicle crashworthiness, which focuses on a vehicle's structural capacity to control crash energy and shield occupants from harm in the event of an incident. Physical crash testing are a major component of traditional crashworthiness evaluation techniques, and they are costly and time-consuming. Computer-Aided Engineering (CAE) has emerged as a key component in the design and assessment of vehicle safety with the development of sophisticated simulation tools. Disparities between the actual test results and the predictions of the CAE model, however, present serious difficulties. By combining them with actual crash test data, Machine Learning (ML) presents a viable way to improve the accuracy of CAE models in this situation.

![Vehicle Crashworthiness](_pages\images\Vehicle_Crashworthiness.png "Machine learning enabled fusion of CAE data and test data for vehicle crashworthiness performance evaluation by analysis")

<center>Figure 1: Vehicle crashworthiness design: Physical test and CAE model </center> 
<center>(Source:https://link.springer.com/article/10.1007/s00158-023-03553-5)</center> 






Motivation: Why This Matters
======
Addressing the gap between CAE predictions and real-world crash test results is crucial for ensuring occupant safety while reducing development costs. Machine Learning models can handle the nonlinearity and missing physics in CAE models, thereby minimizing the reliance on costly physical crash tests. This not only accelerates the certification process but also enhances the robustness of vehicle safety designs.





Why is this Problem Important?
======
**1. Occupant Safety:** Accurate crashworthiness predictions are crucial for ensuring the safety of vehicle occupants.

**2. Cost Reduction:** Physical crash tests are expensive and resource-intensive. Reducing the number of required tests can significantly lower development costs.

**3. Regulatory Compliance:** Meeting safety regulations efficiently is essential for bringing new vehicles to market quickly.




Challenges: The Roadblocks to Accuracy*
======
**Economic Barriers:**
Physical crash tests are expensive and resource-intensive, making them a significant barrier in the development process.

**Data Availability:**
Limited real-world crash test data is available for training machine learning models, which can affect the accuracy and reliability of these models.

**Accuracy:**
Discrepancies between CAE simulation outputs and real-world test results can lead to inaccurate predictions, potentially compromising safety.

**Complexity:**
Non-linear crash dynamics are challenging to model accurately, requiring sophisticated methods to capture the complexities involved.




Mapping to ML: Bridging the Gap
======
To address these challenges, we map the problem to a machine learning framework. The inputs include crash speed, vehicle front and rear weights, while the outputs are deceleration predictions over time or displacement. The loss function aims to minimize the discrepancy between CAE predictions and test data. The goal is to train low-fidelity models using CAE data and fine-tune them with limited high-fidelity crash test data.


**Inputs and Outputs: The Data Dynamics**

1. Inputs: Crash speed, vehicle front weight, vehicle rear weight.
2. Outputs: Deceleration prediction over time or displacement.


**Loss Function: The Precision Metric**

1. Objective: Minimize the discrepancy between CAE predictions and test data.
2. Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE).




Methodology: The Dual Approach 
======

***Time-Domain Approach: The Power of Temporal Convolutional Networks (TCN)***
-----
The time-domain approach leverages Temporal Convolutional Networks (TCN) to model the deceleration response over time. TCN is a type of neural network that uses convolutional layers to process sequential data. It is particularly effective for time-series data due to its ability to capture long-range dependencies and its efficiency in training.

**The process involves:**

Training a Low-Fidelity TCN: Using a large volume of CAE simulation data, a low-fidelity TCN model is trained.

Fine-Tuning with Transfer Learning: The low-fidelity TCN is then fine-tuned into a multi-fidelity TCN using a small number of crash test data through transfer learning.


**Key Parameters: The Blueprint**
-----

1. Input Size: 4 (time, initial speed, vehicle front weight, vehicle rear weight).

2. Output Size: 1 (deceleration at each time step).

3. Number of Channels: 600.

4. Dilation Factor: 2^i, where i is the layer number.

5. Filter Size: 20.

6. Dropout Rate: 0.05.

7. Optimizer: Adam.

8. Learning Rate: 3×10^−4.

9. Training Epochs: 10,000.

10. Batch Size: 32.

11. Loss Function: Mean Squared Error (MSE).

12. Activation Function: ReLU.

13. Normalization: Batch Normalization.

![TCN Architecture](_pages\images\TCN_Architecture.png "The workflow of ML training and prediction for data fusion")

<center>Figure 2: TCN Architecture</center> 
<center>(Source:https://link.springer.com/article/10.1007/s00158-023-03553-5)</center> 



***Displacement-Domain Approach: The Precision of Gaussian Process Regression (GPR)***
-----
The displacement-domain approach models the vehicle crash as a spring-mass system and uses Gaussian Process Regression (GPR) to correct the nonlinear dynamics. GPR is a probabilistic, non-parametric model that can capture complex relationships in data and provide uncertainty estimates for predictions.

**The steps include:**

1. Modeling the Crash: The vehicle crash is modeled as a spring-mass system.

2. GPR for Bias Correction: A GPR model is trained to capture the unmodeled physics of the nonlinear spring constant.

3. Iterative Integration: The GPR model is integrated with the CAE predictions iteratively to improve accuracy.



**Key Parameters: The Formula for Success**
-----

1. Input Size: 1500 (length of training data).

2. Output Size: 1 (model bias).

3. Kernel: Combination of ConstantKernel and MaternKernel.

4. Optimizer: L-BFGS-B algorithm.

5. Alpha: 1×10^−5.

6. Training Repeats: 10.

7. Loss Function: Mean Squared Error (MSE).

8. Uncertainty Quantification: Monte Carlo Simulations.

9. Prediction Interval: 95% Confidence Interval.

![Spring Mass Model](_pages\images\Spring_mass_model.png "Spring–mass model for vehicle crash test")

<center>Figure 3: Spring-Mass Model for Vehicle Crash Test</center> 
<center>(Source:https://link.springer.com/article/10.1007/s00158-023-03553-5)</center> 


Applications: Beyond the Automotive Industry
======

**Automotive Industry: Accelerating Safety**
-----
The automotive industry stands to benefit significantly from this approach. By accelerating crashworthiness certification and reducing reliance on prototype testing, manufacturers can streamline their development processes and bring safer vehicles to market faster.


**Broader Implications: A Versatile Solution**
-----
The potential applications extend beyond the automotive industry. This methodology can be adapted for aerospace and other structural certifications, offering a versatile solution for improving simulation accuracy across various sectors.





Experiments: The Proof in the Pudding
======

**Experimental Data: The Foundation**

1. CAE Data: 1009 simulation datasets generated using Latin hypercube sampling.

2. Crash Test Data: 11 real-world datasets, limited due to the high costs of physical tests.



**Evaluation Metrics: Measuring Success**

The ISO Validation Metrics, including Corridor, Phase, Magnitude, and Slope scores, were used to evaluate the performance of the proposed methods.

**Key Results: The Breakthrough**

1. Time-Domain Approach: The Multi-Fidelity TCN significantly improved predictions compared to CAE alone.

2. Displacement-Domain Approach: The GPR effectively modeled non-linear dynamics and improved accuracy for unseen configurations.


![Validation Metric](_pages\images\Validation_metric.png "ISO Validation Metric")

<center>Figure 4: ISO Validation Metric</center>
<center>(Source:https://link.springer.com/article/10.1007/s00158-023-03553-5)</center>





Results: The Numbers Speak Louder
======

***Time-Domain Approach: The Power of Prediction***
-----
The Multi-Fidelity TCN demonstrated substantial improvements in prediction accuracy over the Low-Fidelity TCN and raw CAE predictions. Transfer learning effectively incorporated test data, enhancing the model's ability to align with real-world observations [1].





**Quantitative Results: The Hard Numbers**
-----

| Test No. | Time Period (ms) | ISO Score (LF_TCN) | ISO Score (MF_TCN) |
|---------|----------------|-------------------|-------------------|
| 3       | 0-20          | 0.451             | 0.727             |
|         | 0-40          | 0.667             | 0.611             |
|         | 0-60          | 0.753             | 0.595             |
| 5       | 0-20          | 0.363             | 0.863             |
|         | 0-40          | 0.566             | 0.850             |
|         | 0-60          | 0.625             | 0.878             |
| 9       | 0-20          | 0.559             | 0.938             |
|         | 0-40          | 0.615             | 0.916             |
|         | 0-60          | 0.476             | 0.878             |


![Time Domain Approach](_pages\images\Time_Domain_Results.png "Time Domain Approach Results")

<center>Figure 5: Time-Domain Approach Results </center>
<center>(Source:https://link.springer.com/article/10.1007/s00158-023-03553-5)</center>



***Displacement-Domain Approach: The Precision of Probabilistic Predictions***
-----
The GPR captured unmodeled physics, leading to enhanced predictions. Additionally, the model quantified prediction uncertainties using Monte Carlo simulations, providing a probabilistic assessment of crashworthiness [1].


**Quantitative Results: The Hard Numbers**
-----

| Test No. | Time Period (ms) | ISO Score (CAE) | ISO Score (GPR) |
|----------|-----------------|-----------------|-----------------|
| 3        | 0-20            | 0.774           | 0.863           |
| 3        | 0-40            | 0.768           | 0.848           |
| 3        | 0-60            | 0.822           | 0.862           |
| 5        | 0-20            | 0.477           | 0.791           |
| 5        | 0-40            | 0.565           | 0.831           |
| 5        | 0-60            | 0.656           | 0.799           |
| 9        | 0-20            | 0.645           | 0.734           |
| 9        | 0-40            | 0.665           | 0.731           |
| 9        | 0-60            | 0.527           | 0.766           |

![Displacement Domain Approach](_pages\images\Displacement_Domain_Results.png "Displacement Domain Approach Results")

<center>Figure 6: Displacement-Domain Approach Results</center> 
<center>(Source:https://link.springer.com/article/10.1007/s00158-023-03553-5)</center>


Comparison of the Two Methods
======

**Time-Domain vs. Displacement-Domain**
----

***Time-Domain Approach (TCN):***

1. Strengths: Efficient in handling sequential data, captures long-range dependencies, and is computationally efficient.

2. Implications: Suitable for scenarios where the temporal dynamics of the crash are crucial. It provides deterministic predictions and is less sensitive to the configuration of training data.

***Displacement-Domain Approach (GPR):***

1. Strengths: Provides probabilistic predictions with uncertainty quantification, captures complex non-linear relationships, and is more robust to the configuration of training data.

2. Implications: Ideal for scenarios where uncertainty quantification is important. It is more flexible and can handle cases where the training data does not closely match the prediction scenarios.



Which Method is Better?
======

The choice between the time-domain and displacement-domain approaches depends on the specific requirements of the application:

**Time-Domain Approach:** Preferred when the temporal dynamics of the crash are critical and the training data closely matches the prediction scenarios.

**Displacement-Domain Approach:** Preferred when uncertainty quantification is important and the training data may not closely match the prediction scenarios.



Pros and Cons: The Double-Edged Sword
======

***Pros: The Bright Side***
-----
  1. Cost Reduction: Minimizes the need for physical crash tests, reducing development costs.

  2. Improved Accuracy: Enhances prediction accuracy for unseen scenarios, ensuring better safety outcomes.
  
  3. Flexibility: Adaptable across different configurations and speeds, offering a versatile solution.
  
  4. Probabilistic Predictions: The displacement-domain approach provides uncertainty quantification, aiding risk-informed decision-making.


***Cons: The Challenges Ahead***
-----
  1. Computational Resources: Requires significant computational resources for model training.
  
  2. Data Limitations: Limited availability of high-fidelity crash test data can affect model performance.
  
  3. Complexity: The methods require expertise in both machine learning and vehicle dynamics.




Conclusion: The Future of Crashworthiness
======
Machine Learning models effectively bridge the gap between CAE predictions and test data, offering a powerful tool for improving vehicle crashworthiness. The time-domain and displacement-domain approaches complement each other, providing robust solutions for enhancing simulation accuracy. Future work will focus on expanding datasets for diverse crash scenarios and exploring hybrid ML models for faster training and better accuracy.




Future Work: The Road Ahead
-----
  1. Dataset Expansion: Incorporate more diverse crash scenarios to improve model robustness.
  
  2. Hybrid Models: Explore hybrid ML models combining the strengths of different approaches.
  
  3. Real-Time Applications: Develop models capable of real-time predictions during crash simulations.

Thank you for reading! If you have any questions or feedback, feel free to reach out.


References
======

1. Zeng, J., Li, G., Gao, Z., Li, Y., Sundararajan, S., Barbat, S., & Hu, Z. (2023). Machine learning enabled fusion of CAE data and test data for vehicle crashworthiness performance evaluation by analysis. Structural and Multidisciplinary Optimization, 66(1), 96. https://doi.org/10.1007/s00158-023-03553-5



