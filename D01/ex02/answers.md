# Exercise 02 - AI Key Notions

## 1 - When we pre-process the training examples, why are we adding a column of ones to the left of the x vector (or X matrix) when we use the linear algebra trick?

To simply "add" theta0 as bias instead of weight (theta0*1 + theta1+feature1 + theta2*feature2 +...)

## 2 - Why does the cost function square the distance between the data points and their predicted values?

To penalize harder big errors

## 3 - What does the cost function value represent?

how far off is our prediction

## 4 - Toward which value would you like the cost function to tend to? What would it mean?

0.0 would mean that our model perfectly fit the truth values
