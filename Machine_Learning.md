
Underfitting: too little model capacity

Overfitting: too much model capacity

How much do we care about explain ability?

How much do we care about making predictions?

<h1>Part 1: Getting ready</h1>
<h2>Know your data</h2>
Summary statistics
<ol>
  <li>Percentiles can help identify the range for most of the data. </li>
  <li>Averages and medians can describe central tendency. </li>
  <li>Correlations can indicate strong relations </li>
</ol>
Visualize the data
<ol>
  <li>Box-plots can help identify outliers </li>
  <li>Density plots and histograms show the spread of the data </li>
  <li>Scatter plots can describe bivariate relationships </li>
</ol>
<h2>Clean your data</h2>
Deal with missing values
<ol>
  <li>Missing data effects some models more than others </li>
  <li>Even for models that can handle missing data, they can be sensitive to it (missing data for certain variables can lead to poor predictions).</li>
  <li>Missing data can be more common in production </li>
  <li>Missing value imputation can get very sophisticated </li>
</ol>
Choose what to do with outliers
<ol>
  <li>An outlier is somewhat subjective. </li>
  <li>Outliers can be very common in multidimensional data </li>
  <li>Some models are less sensitive (ex. tree models are more robust) to outliers than others (ex. Regression models are less robust) </li>
  <li>Outliers can be the result of bad data collection, or they can be legitimate extreme (or unusual) values. </li>
  <li>Sometimes outliers are the interesting data points we want to model, and other times they just get in the way. </li>
</ol>
Does the data need to be aggregated?
<ol>
  <li>Row data is somewhat too granular for modeling. </li>
  <li>The granularity of the data affects the interpretation of our model </li>
  <li>Aggregating data can also remove bias posed by more frequent observations in the raw data </li>
  <li>Aggregating data can also lessen the number of missing values and the effect of outliers </li>
</ol>
<h2>Augment the data</h2>
Feature engineering is the process of going from raw data (ex. Date-time) to data that is ready for modeling (ex. Day of the week or month). It can serve multiple purposes:
<ol>
  <li>Make the model easier to interpret (ex. binning) </li>
  <li>Capture more complex relationships (ex. NNs) </li>
  <li>Reduce data redundancy and dimensionality (ex. PCA) </li>
  <li>Rescale variables (ex. Standardizing or normalizing) </li>
</ol>
Different models may have different feature engineering requirements. Some have built-in feature engineering.



<h1>Part 2: Choosing and tuning models</h1>
We have our data, an idea of the models we want to build, we completed part 1 and are ready to do some modeling. We have to train (use 75% of data to build model) and test (use 25% to evaluate model) our models. A good model should be able to make predictions on data it wasn't trained with.
<h2>What is a model?</h2>
At a high level, a model is a simplification of something more complex.
"All models are wrong, but some are useful." - statistics mantra
A machine learning algorithm uses data to automatically learn the rules. It simplifies the complexities of the data into relationships described by the rules. A predictive model is an algorithm that learns the prediction rules.
<h2>What is a good predictive model?</h2>
Specifics of the model itself
<ol>
  <li>Accurate: Are we making good predictions? </li>
  <li>Interpretable: How easy is it to explain how the predictions are made? </li>
</ol>
The way the model is being deployed
<ol>
  <li>Fast: How long does it take to build a model, and how long does the model take to make predictions?  </li>
  <li>Scalable: How much longer do we have to wait if we build/predict using a lot more data? </li>
</ol>
<h2>What is model complexity?</h2>
A model is more complex if:
<ol>
  <li>It relies on more features to learn and predict (ex. 2 vs 10 features to predict a target) </li>
  <li>It relies on more complex feature engineering (ex. Polynomial terms, interactions, or principle components) </li>
  <li>It has more computational overhead (ex. A single decision tree vs random forest of 100 trees) </li>
</ol>
Opposite of explain ability
Computational overhead
<h2>What is model complexity across the same model?</h2>
The same ML algorithm could be made more complex based on the number of parameters or the choice of some hyper-parameters
<ol>
  <li>A regression model can have more features, or polynomial terms and interaction terms </li>
  <li>A decision tree can have more or less depth </li>
</ol>
Making the same algorithm increases the chance of overfitting
"Models have to be simple, but not simplistic" - Einstein
<h2>What is model complexity across different models?</h2>
Complexity can also refer to choosing a more complex algorithm, which is generally less efficient (to train and to predict), in addition to being less easy to understand.
<ol>
  <li>A neural network is similar to regression but much more complex in its feature engineering </li>
  <li>A random forest is similar to a decision tree, but complex because it builds multiple trees </li>
</ol>



<h1>Part 3: Deploying Models</h1>
We have a model and we are ready to use it.
<h2>Model consumption</h2>
Its important to know as much as possible how models are to be consumed
<ol>
  <li>A model that is consumed by a web app needs to be fast </li>
  <li>A model that is used to predict in batch needs to be scalable </li>
  <li>A model that updates a dashboard as data streams in may need to be fast and scalable </li>
  <li>Is this a good model in terms of accuracy? </li>
</ol>



<h1>Part 4: The data science lifecycle</h1>


<h2>Lifecycle</h2>
<ol>
  <li>Define objective: What problem am I solving?</li>
  <li>Collect and manage data: What information do I need?</li>
  <li>Build the model: find patterns in the data that leads to a solution.</li>
  <li>Evaluate and critique the model: Does the model solve my problem?</li>
  <li>Present results and document: Establish that I can solve the problem, and how.</li>
  <li>Deploy the model: Doeploy the model to solve the problem in the real world.</li>
</ol>
<h2>The choice of a model affects and is affected by:</h2>
<ol>
  <li>Whether the model meets the business goals</li>
  <li>How much pre-processing the models needs</li>
  <li>How accurate the model is</li>
  <li>How explainable the model is</li>
  <li>How fast the model is in making predictions</li>
  <li>How scalable the models is (building and predicting)</li>
</ol>

