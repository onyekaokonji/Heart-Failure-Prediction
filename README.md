# Heart-Failure-Prediction
A logistic classification algorithm for the detection of heart failure.

### Libraries used
SckitLearn, Pandas, Matplotlib

### Details of dataset
Dataset was an imbalanced one such that the number of positive cases is way less than the negative cases to the ratio 1:2. However, considering the cases of heart failure in a random population is hardly ever in the majority, the datasets weren't balanced as this will mess with real-life performance of the algorithm.

### Performance 
<img width="480" alt="Screenshot 2021-11-06 at 17 08 27" src="https://user-images.githubusercontent.com/40724187/140616193-382527ef-2006-4e0d-9193-f92ed01fff9e.png">

### Web link to view performance:
http://192.168.8.100:8501

### Possible way forward
Because it's a health-related algorithm, it's sensible to want to improve recall to somewhere around upwards of 90%. Possible methods might be to:
<ol type: 'i'> 
  <li>change the threshold of the decision function of the classifier</li>
  <li>changing the algorithm all together</li>
  <li>adding new features</li>
  <li>upsampling the positive cases</li>
</ol>
