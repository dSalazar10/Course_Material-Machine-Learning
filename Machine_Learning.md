Model Development Lifecycle:
- Define objective: What problem am I solving?
- Collect and manage data: What information do I need?
- Build the model: find patterns in the data that leads to a solution.
- Evaluate and critique the model: Does the model solve my problem?
- Present results and document: Establish that I can solve the problem, and how.
- Deploy the model: Doeploy the model to solve the problem in the real world.

Underfitting: too little model capacity

Overfitting: too much model capacity


Part 1: Getting ready

Know your data
-	Summary statistics
<ol>
  <li>Percentiles can help identify the range for most of the data. </li>
  <li>Averages and medians can describe central tendency. </li>
  <li>Correlations can indicate strong relations </li>
</ol>
-	Visualize the data
<ul>
<li> </li>
</ul>
o	Box-plots can help identify outliers
o	Density plots and histograms show the spread of the data
o	Scatter plots can describe bivariate relationships
Clean your data
-	Deal with missing values
<ul>
<li> </li>
</ul>
o	Missing data effects some models more than others
o	Even for models that can handle missing data, they can be sensitive to it (missing data for certain variables can lead to poor predictions).
o	Missing data can be more common in production
o	Missing value imputation can get very sophisticated
-	Choose what to do with outliers
<ul>
<li> </li>
</ul>
o	An outlier is somewhat subjective.
o	Outliers can be very common in multidimensional data
o	Some models are less sensitive (ex. tree models are more robust) to outliers than others (ex. Regression models are less robust)
o	Outliers can be the result of bad data collection, or they can be legitimate extreme (or unusual) values.
o	Sometimes outliers are the interesting data points we want to model, and other times they just get in the way.
-	Does the data need to be aggregated?
<ul>
<li> </li>
</ul>
o	Row data is somewhat too granular for modeling.
o	The granularity of the data affects the interpretation of our model
o	Aggregating data can also remove bias posed by more frequent observations in the raw data
o	Aggregating data can also lessen the number of missing values and the effect of outliers
Augment the data
-	Feature engineering is the process of going from raw data (ex. Date-time) to data that is ready for modeling (ex. Day of the week or month). It can serve multiple purposes:
<ul>
<li> </li>
</ul>
o	Make the model easier to interpret (ex. binning)
o	Capture more complex relationships (ex. NNs)
o	Reduce data redundancy and dimensionality (ex. PCA)
o	Rescale variables (ex. Standardizing or normalizing)
-	Different models may have different feature engineering requirements. Some have built-in feature engineering.



Part 2: Choosing and tuning models

We have our data, an idea of the models we want to build, we completed part 1 and are ready to do some modeling. We have to train (use 75% of data to build model) and test (use 25% to evaluate model) our models. A good model should be able to make predictions on data it wasn't trained with.

What is a model?
-	At a high level, a model is a simplification of something more complex.
-	"All models are wrong, but some are useful." - statistics mantra
-	A machine learning algorithm uses data to automatically learn the rules. It simplifies the complexities of the data into relationships described by the rules. A predictive model is an algorithm that learns the prediction rules.
What is a good predictive model?
-	Specifics of the model itself
<ul>
<li> </li>
</ul>
o	Accurate: Are we making good predictions?
o	Interpretable: How easy is it to explain how the predictions are made?
-	The way the model is being deployed
<ul>
<li> </li>
</ul>
o	Fast: How long does it take to build a model, and how long does the model take to make predictions? 
o	Scalable: How much longer do we have to wait if we build/predict using a lot more data?
What is model complexity?
-	A model is more complex if:
<ul>
<li> </li>
</ul>
o	It relies on more features to learn and predict (ex. 2 vs 10 features to predict a target)
o	It relies on more complex feature engineering (ex. Polynomial terms, interactions, or principle components)
o	It has more computational overhead (ex. A single decision tree vs random forest of 100 trees)
-	Opposite of explain ability
-	Computational overhead
What is model complexity across the same model?
-	The same ML algorithm could be made more complex based on the number of parameters or the choice of some hyper-parameters
<ul>
<li> </li>
</ul>
o	A regression model can have more features, or polynomial terms and interaction terms
o	A decision tree can have more or less depth
-	Making the same algorithm increases the chance of overfitting
-	"Models have to be simple, but not simplistic" - Einstein
What is model complexity across different models?
-	Complexity can also refer to choosing a more complex algorithm, which is generally less efficient (to train and to predict), in addition to being less easy to understand.
<ul>
<li> </li>
</ul>
o	A neural network is similar to regression but much more complex in its feature engineering
o	A random forest is similar to a decision tree, but complex because it builds multiple trees
How much do we care about explain ability?
How much do we care about making predictions?


Part 3: Deploying Models

We have a model and we are ready to use it.
Model consumption
-	Its important to know as much as possible how models are to be consumed
<ul>
<li> </li>
</ul>
o	A model that is consumed by a web app needs to be fast
o	A model that is used to predict in batch needs to be scalable
o	A model that updates a dashboard as data streams in may need to be fast and scalable
o	Is this a good model in terms of accuracy?

Part 4: The data science lifecycle
The choice of a model affects and is affected by:
-	Whether the model meets the business goals
-	How much pre-processing the models needs
-	How accurate the model is
-	How explainable the model is
-	How fast the model is in making predictions
-	How scalable the models is (building and predicting)
