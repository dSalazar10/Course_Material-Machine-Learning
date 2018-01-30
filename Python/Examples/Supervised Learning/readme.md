Given examples of inputs and the desired categories, predict the outputs of future samples.
The aim is to learn a mapping from the input to an output whose correct values are provided by a supervisor 

Examples:
1. Learning associations (Basket analysis):

Finding associations between products purchased by customers. If people who by X typically buy Y, and if there is a customer who buy X but doesn't buy Y, they are a Y customer. We can target them for cross selling, otherwise known as an association rule. Find the conditional probability P(X|Y) where Y is the product we would like to condition on X, which is the product or set of products which we know the customer has already purchased. For example: if P(beer|chips) = 0.7, then we could say 70% of people who buy beer also buy chips. We might want to also condition the customer's attributes along with Y, P(X|Y,D) where D is the set of customer attributes. 

Store: Products -> Cross selling

Bookseller: Books -> Cross selling

Web Portal: link clicks -> Cache pages in advanced

2. Classification (Credit Scoring):

Determining whether to give a loan to a customer or not. The bank has info on the customer: income, savings, collaterals, profession, age, and past financial history. The bank has a record of past loans containing such customer data and whether the loan was paid back or not. The aim is to infer a general rule coding the association between a customer's attributes and his risk. The machine learning system fits a model to the past data to be able to calculate the risk for a new application and then decides to accept or refuse it accordingly. The features are the customer info and the classes are paid or not. Create a model that maps the features/classes to either low-risk and high-risk customers. 
For example: customer = (income > i[1] && savings > i[2])? low-risk : high-risk;
Alternatively, instead of having only a low or high risk output, you could determine the probability, P(Y|X) where X is the customer attributes and Y is 0/1, of the customer paying back the loan. Then for a given X = x, if we have P (Y = 1|X = x) = 0.8, we say that the customer has an 80 percent probability of being high-risk, or equivalently a 20 percent probability of being low-risk. 

3. Pattern Recognition:
- Handwritten digit recognition
- Spelling correction
- Facial recognition
- Medical diagnosis
- Speech recognition
- Spam filtering
- Trending topic detection
- Machine translation
- Biometric authentication
- Knowledge extraction
- Compression
- Outlier detection

4. Regression

Predict the price of a used car. Inputs are the car attributes: brand, year, engine capacity, and mileage. The output is the price of the car. machine learning program fits a function to this data to learn Y as a function of X. Create a model that learns the mapping from the input X to the output Y. Assume a model defined up to a set of parameters y = g(x| ) where g() is the model and   are its parameters. Y is a number in regression and is a class code (0/1) in the case of classification. The machine learning program optimizes the parameters, θ, such that the approximation error is minimized, that is, our estimates are as close as possible to the cor- rect values given in the training set. To decrease under fitting, quadratic or higher-order polynomial may be used, optimizing its parameters for best fit. 

Another example of regression is navigation of a mobile robot, for ex- ample, an autonomous car, where the output is the angle by which the steering wheel should be turned at each time, to advance without hitting obstacles and deviating from the route. Inputs in such a case are pro- vided by sensors on the car-for example, a video camera, GPS, and so forth. Training data can be collected by monitoring and recording the actions of a human driver. 

Another example is response surface design: we want to build a machine that roasts coffee. The machine has many inputs that affect the quality: various settings of temperatures, times, coffee bean type, and so forth. We make a number of experiments and for different settings of these inputs, we measure the quality of the coffee, for example, as consumer satisfaction. To find the optimal setting, we fit a regression model linking these inputs to coffee quality and choose new points to sample near the optimum of the current model to look for a better configuration. We sample these points, check quality, and add these to the data and fit a new model. 

