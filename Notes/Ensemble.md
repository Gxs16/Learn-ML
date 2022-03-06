# Ensemble

## Bagging

* Bagging trains n base learners in parallel
* Make decisions by averaging learners' outputs (regressiong) or majority boting(classification)
* Each Learner is trained on data by bootstrap sampling
  * Given a dataset of examples, create a sample by randomly sampling examples with replacement
  * Around $1-\lim (1-1/m)^m=1-1/e$ unique examples will be sampled, use the out-of-bag examples for validation
* Bagging recudes variance, especially for unstable learners. Bagging reduces more variance when base learners are unstable
* Groud truth: $f$, Base learner $h$, Bagging: $\hat{f}=E[h(x)]$
* Given $(E[x])^2 \leq E[x^2]$:
$$
(f(x)-\hat{f}(x))^2 \leq E[(f(x)-h(x))^2] \Leftrightarrow (E[h(x)])^2 \leq E[h(x)^2]
$$

### Random Forest

* base learner = decision tree
* Often randomly select a subset of features for each learner

## Boosting

* Boosting combines week learners into a strong one to reduce bias
* Learn n weak learners sequentially, at step $i$:
  * Train a weak learner $h_i$, evaluate its errors $\epsilon_i$
  * Re-sample data according to $\epsilon$ ro focus on wrongly predicted samples

### Gradient Boosting

* Supports arbitrary differentiable loss
* $H_t(x)$: output of combined model at timestep $t$, with $H_1(x) = 0$
* For each step $t$:
  * Train a new learner $\hat f_t$ on residuals: $\{(x_i, y_i-H_t(x_i))\}_{i=1, ... m}$
  * $H_{t+1}(x)=H_t(x)+\eta\hat f_t(x)$
    * The learning rate $\eta$ regularizes the model by shrinkage
* MSE:$L = \frac{1}{2}(H(x)-y)^2$, The residuals equal to $-\partial L/\partial H$
* For other loss $L$, learner $\hat f_t = \argmin \frac{1}{2}(\hat f_t(x)+\partial L/\partial H_t)^2$
* Avoid overfitting: subsampling, shrinkage, early-stopping

### Adaboost

* Initialize the data weighting coefficients $\{w_n\}$ by setting $w_n^{(1)} = 1/N $ for $n=1, ..., N$
* For $m=1, ..., M$:
  * Fit a classifier $y_m(x)$ to the training data by minimizing the weighted error function
$$J_m=\sum_{n=1}^N w_n^{(m)}I(y_m(X_n) \neq t_n)$$
where $I(y_m(x_n) \neq t_n)$ is the indicator function and equals 1 when $y_m(x_n) \neq t_n$ and 0 otherwise.
  * Evaluate the error rate:
$$\epsilon_m=\frac{\sum_{n=1}^N w_n^{(m)}I(y_m(X_n) \neq t_n)}{\sum_{n=1}^N w_n^{(m)}}$$
and then use these to evaluate $$\alpha_m=\ln\{\frac{1-\epsilon_m}{\epsilon_m}\}$$
  * Update the data weighting coefficients $$w_n^{m+1}=w_n^{m}\exp\{\alpha_mI(y_m(x_n)\neq t_n)\}$$
* Make predictions using the final model, which is given by $$Y_M(x)=sign(\sum_{m=1}^M\alpha_my_m(x))$$

## Stacking

* Combine multiple base learners to reduce variance

### Multi-Layer Stacking

* Stacking base learners in multiple levels to reduce bias
  * Can use a different set of base learners at each level
* Upper levels (e.g. L2) are trained on the outputs of the level below (e.g. L1)
  * Concatenating original inputs helps
* Overfitting
  * Train leaners from different levels on different data to alleviate overfitting 
    * Split training data into A and B, train L1 learners on A, run inference on B to generate training data for L2 learners
  * Repeated k-fold bagging
    * Train k models as in k-fold cross validation
    * Combine predictions of each model on out-of-fold data
    * Repeat step 1,2 by n times, average the predictions of each example for the next level training
