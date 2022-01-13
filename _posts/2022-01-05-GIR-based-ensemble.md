---
title: GIR-based ensemble sampling approaches for imbalanced learning
categories:
 - Statistics
tags:
 - Machine Learning
 - Imbalanced data 

use_math: true

last_modified_at: 2022-01-05T19:19:00-23:00
---

<br/>

_(2017)GIR-based ensemble sampling approaches for imbalanced learning - Bo Tang, Haibo He_

<br/>

## Introduction 

이 논문은 **imbalanced learning**을 다루기 때문에 먼저 **imbalance하다는 것은 무엇인가**에 대해서 먼저 정의할 필요가 있다. imbalance하다는 것은 _"the class distribution is uneven for a given imbalanced data set_", 즉 주어진 데이터의 class 분포가 동일하지 않다는 것을 의미한다. 그럼 이제 이게 왜 중요하고 issue가 되는가에 대해 생각해볼 수 있는데, 먼저 생각해볼 점은 기존의 머신러닝 모형은 balanced data set을 생각하고 만들어진 모형이라는 점이다. 이 모형으로 학습을 하게 되면 하나의 많은 class(majority)가 적은 class(minority)를 dominate하는 현상이 발생한다. 이렇게 되면 majority에 대해서 학습이 잘 되기만 하면 모형은 학습이 잘 된 것으로 판단한다. 예를 들어서 majority class: minority class가 90:10이라면 90에 대해서만 잘 분류해도 정확도가 90%라는 좋은 성과를 내게 된다. 하지만 우리는 minority에 관심이 가고 가치있는 경우가 많다. 

두번째로 생각해볼 수 있는 점은 실제로 imbalance한 데이터인 경우가 굉장히 많기 때문이다. 의료 데이터나 사고, 마케팅, 정부 정책 등에 대한 데이터를 보면 minority가 더 값진 경우가 많다. 암에 걸린 환자인지 아닌지 분류하고자 한다고 하면 암이 아닌 환자가 훨씬 많을 것 아닌가. 따라서 현실적으로도 imbalance data에 대해서 어떻게 학습을 할 것인지는 중요한 연구라고 할 수 있다. 

위에 두 가지 측면으로 제시한 점에서 뽑아낼 수 있는 우리의 목표는 결국 **majority에 대한 최소의 cost를 희생하면서 minority의 분류 성능을 높이는 것**으로 세울 수 있다. 

이 문제를 해결하기 위해서 많은 논문들이 나왔는데 그 중에서 SMOTE(Systhetic Minority Over-sampling TEchnique)나 ADASYN(ADAptive SYNthetic sampling approach), ensemble-based undersampling등이 효과적이고 간단한 기법으로 알려져 있다. 이런 sampling based approach에서 가장 중요한 이슈 중에 하나는 '어떻게 새롭게 만들 데이터셋의 가장 좋은 balance를 맞출 것인가'이다. 대부분의 approach는 majority와 minority 사이의 class imbalance를 측정하기 위해 **sample size ratio**를 사용한다. 그렇게 해서 majority와 minority의 샘플 개수를 맞춰 주는데, 이렇게 맞춰줘도 **class imbalance는 아직 남아 있을 수 있다!** 이 점을 해결하려고 이 논문이 나온건데, 이 논문에서는 class imbalance의 새로운 measure 방법으로 **Generalized Imbalance Ratio(GIR)**을 도입한다. 

이것말고도 생길 수 있는 문제가 하나 더 있다. 일반적으로 oversampling을 하거나 undersampling을 하면 별로 의미 없는 minority class가 생성되거나 의미 있는 majority class가 버려지면서 classfier의 성능이 저하되는 현상이 발생하곤 하는데, 이 논문에서는 불균형 학습 문제를 undersampling이나 oversampling을 써서 여러 개의 balanced subproblem으로 바꾸고, classfier들이 잘 분류하지 못하는 샘플들에 더 집중할 수 있게 만든 다음, 최종적으로 strong classifier를 세우는 ensemble 기법을 사용하면서 해결한다. 그리고 GIR을 써서 undersampling을 하는지 oversampling을 하는지에 따라서 GIREnUS와 GIREnOS로 나눠서 설명한다. 

결국 논문의 키워드를 정리하면 다음과 같이 세 가지로 나타낼 수 있다. 

- GIR, which measures class distribution imbalance between two classes, based on their intra-class coherence
- Adaptively split the imbalanced learning problem into multiple balanced learning subproblems
- GIREnUS, GIREnOS



<br/>

## GIR: Generalized Imbalance Ratio 

<p align="center">
  <img src="/assets/images/GIR-based_ensemble/Fig2.jpeg" width=500>
</p>

### Motivation

introduction에서도 언급했지만 기존 모형은 imbalance 정도를 측정하기 위해 sample size ratio를 활용하는데, sample size ratio는 class distribution에서의 imbalance 정도를 알려주지 않는다! 위의 예제를 통해서 확인해보자. (a)의 경우에는 Class1이 minority, Class2가 majority이고, (b)의 경우에는 sample size ratio가 비슷하지만 class-imbalanced한 데이터이다. 

(a)에서는 imbalance한 경우이지만 Class1과 Class2가 쉽게 구분될 수 있을 거라는 것은 직관적으로 알 수 있다. 두 class 간의 겹치는 부분이 없기 때문에 classfication boundary를 정하는 것이 매우 쉽다. 

(b)의 경우는 두 sample의 sample size ratio가 거의 동일함에도 class1의 sample들이 더 집중되어 있기 때문에 두 class를 분할하는 경계를 결정하는데 class2의 영향이 적게 미치게 된다. 따라서 Class2에 대한 분류 성능이 떨어지는 결과를 낳는다. 예를 들어서 Class1을 완벽하게 분류하는 경계선을 긋는다면 Class2에 큰 영향을 받지 않고 분류 정확도가 높게 나올 것이다. 이것은 우리가 원하는 경우가 아니기 때문에 sample size ratio를 imbalance의 정도를 측정하는 수단으로 사용하는 것은 문제가 있다. 따라서 이 논문에서는 기존 연구에서 사용하던 "majority/minority" 용어 대신에 "positive/negative"라는 용어를 대신 사용한다. 조금 다른 의미인데, majority랑 minority는 sample ratio를 비교해서 나누는 것이라면 positive랑 negative는 분포가 dominate한지에 따라 나뉜다.(negative가 dominate, positive가 dominated)

<br/>

### GIR definition 

* training data set:  
  $\chi = \{ (\textbf{x}_1, y_1), (\textbf{x}_2, y_2), \cdots, (\textbf{x}_N, y_N) \}$
  
  _where_ $\textbf{x}_i \in R^d $ and $y_i \in \{+1, -1\}$


* the set of positive samples: $\mathcal{P}$ , sample size: $N_{+}$

* the set of negative samples: $\mathcal{N}$, sample size: $N_{-}$



논문에서는 generalized class-wise statistic을 사용해서 intra-class coherence(클래스 내 일관성)을 측정하는데, generalized class-wise statistic은 k-NN을 활용한 ENN(Extended Nearest Neighbor)에서 제안된 방법이다. 

>  ENN에 대해서 간략하게 설명하자면, ENN은 intra-class coherence의 최대치를 기반으로 test sample의 class를 예측하는데 k-NN과 달리 test sample에서 가장 가까운 neighbor만 탐색하는 것이 아니라 test sample 자체를 가장 가까운 neighbor로 간주하는 sample도 고려하는 방식을 사용한다. 모든 training data로부터 generalized class-wise statistic을 사용해서 전체 분포에 대해 학습할 수 있다. 

The generalized class-wise statistic for the majority class:  

$$
\begin{align}
T_{+} &= \frac{1}{N_+}\sum_{\textbf{x} \in \mathcal{P}}\frac{1}{k}\sum_{r=1}^{k}I_r(\textbf{x}, \chi) \\
&= \frac{1}{N_+}\sum_{\textbf{x} \in \mathcal{P}}t_k(\textbf{x})
\end{align}
$$



- $k$: total number of nearest neighbors to be considered
- $I_r(\mathbf{x}, \chi)$: the indicator function indicating wheter data sample $\textbf{x}$ and its $r$th nearest neighbor in $\chi$, denoted by $NN_r(\mathbf{x}, \chi)$ are from the same class or not
- $t_k(\textbf{x})$: a point-wise statistic for the sample $\textbf{x}$ which evaluates how many samples in its $k$ nearest neighbors come from its own class



The generalized class-wise statistic for the minority class:  



$$T_{-} = \frac{1}{N_{-}}\sum_{\textbf{x} \in \mathcal{N}}t_k(\textbf{x})$$


이를 정리하면, $T_{+}, T_{-}$는 각각 positive class, negative class에 대해서 intra-class coherence를 구하는 것이고, 이 말은 곧 한 클래스 내에 속하는 샘플들의 가장 가까운 이웃들이 다른 클래스에 있는 샘플들에 의해서 얼마나 dominate되어 있는지를 측정하는 척도라고 볼 수 있다. 예를 들어서 $T_{+}$가 큰 값을 갖는다면 positive sample이 concentrated되어 있고, 가장 가까운 이웃들이 positive sample에 의해 dominate되어 있다는 뜻이다. 반대로 $T_{+}$가 작은 값을 갖는다면 이웃들이 negative sample에 의해서 dominate 되어 있다는 뜻이다. 

> 한 마디로 정리: $T_{+}$가 크다? $\rightarrow$ positive sample들이 뭉쳐있다. 

<br/>

### Theoretical Properties

이 논문에서는 GIR이 갖는 이론적 특징들을 수학적으로 증명하고 있지만 여기서는 증명을 다루지는 않도록 한다. (중요한 증명이긴 하지만 해석학에 대한 공부가 더 필요함)

1. 두 데이터셋이 balanced sample size ratio를 가질 때 $\lim_{N \rightarrow \infty}E(\Delta T) = 0$ 가 성립한다. 
2. 두 데이터셋이 balanced sample ratio를 성립하면서 무수히 많은 샘플 수를 가진다면 GIR은 sample size ratio와 같다. 

<br/>

## GIR-based ensemble sampling approaches


### GIREnUS: GIR-based ensemble undersampling approach

이 부분부터는 다른 알고리즘과 거의 유사한 undersampling을 보여주는데, sampling probability에 주목할 필요가 있다. 

<p align="center">
  <img src="/assets/images/GIR-based_ensemble/GIREnUS_algorithm.jpeg" width=500>
</p>

 sampling probability는 다음과 같다.

 $p_{-}(\mathbf{x}) = \frac{t_{k}(\mathbf{x}) + 1}{\sum_{\mathbf{x} \in \mathcal{N}}t_{k}(\mathbf{x}) + N_{-}}$ 



이것을 해석해보면, 'negative class에 속하는 sample들이 뭉쳐있는 것 대비 그 점은 얼마나 negative class로 뭉쳐있는가''이다. 분모와 분자에 $N_{-}$와 $1$을 더해주는 것은 분모와 분자가 0이 되는 것을 방지하기 위함이라고 생각하면 된다. 따라서 $p_{-}(\mathbf{x})$가 크다는 것은 sample이 학습되기 쉽다는 것이고, 굳이

알고리즘을 요약하면,

1. $\Delta T \leq 0$, 즉 class imbalance가 해소될 때까지 학습하기 쉬운 negative sample 제거 
2. 1번 과정이 끝나서 balanced data set이 형성되면 $Adaboost_{1}$ 학습 
3. 총 $U$번의 과정으로 $Adaboost_{1}, \cdots, Adaboost_{U}$ 생성 
4. $Adaboost_1, \cdots, Adaboost_U$를 bagging하여 final classifier 생성



<br/>

### GIREnOS: GIR-based ensemble oversampling approach

<p align="center">
  <img src="/assets/images/GIR-based_ensemble/GIREnOS_algorithm.jpeg" width=500>
</p>

GIREnUS와 거의 동일하지만 undersampling, oversampling의 차이이다. oversampling probability는 다음과 같다. 

$p_{+}(\mathbf{x}) = 1- \frac{t_k(\mathbf{x}) + 1}{\sum_{\mathbf{x} \in \mathcal{P}}t_k(\mathbf{x}) + N_{+}}$

이번에는 학습이 쉬운 점이 아니라 학습이 어려운 점에 가중치를 줘서 sampling을 해야 하기 때문에 1에서 뺀 값을 sampling probability로 취한다. 



1. $\Delta T \leq 0$, 즉 class imbalance가 해소될 때까지 학습하기 어려운 positive sample 생성 
2. 1번 과정이 끝나서 balanced data set이 형성되면 $Adaboost_{1}$ 학습 
3. 총 $U$번의 과정으로 $Adaboost_{1}, \cdots, Adaboost_{U}$ 생성 
4. $Adaboost_1, \cdots, Adaboost_U$를 bagging하여 final classifier 생성



와 같이 알고리즘을 요약할 수 있다. 

<br/>



## Complexity Analysis

Experiment는 자신들의 알고리즘이 F-measure, G-mean, AUC 등의 척도에서 우수한 성능을 보임을 입증한다. 

time complexity는 GIR을 계산하기 위해 nearest neighbor를 모두 탐색해야 하기 때문에 매우 높을 것으로 예상했는데 논문에서는 무시 가능한 수준이라고 한다. 그 이유는 nearest neighbor에 대한 통계량을 구할 때 KNN graph를 활용하여 pre-calculate하기 때문에 다시 계산할 필요가 없다는 것이다. 물론 아주 간단한 데이터 샘플링 모델과는 complexity 차이가 있긴 하지만 복잡한 모델과 비교했을 때는 생각보다 괜찮은 time complexity를 보였다. 



<br/>



## Conclusion 

기존에 사용하던 imblance measure인 sample size ratio의 한계를 지적하고 그에 대한 대안으로 GIR을 제시한 점에서 contribution이 있다고 생각한다. 하지만 iteration마다 balanced data set을 형성하고 Adaboost를 학습할 때 hyperparameter 조정이 문제가 될 것으로 예상된다. (Boosting은 hyperparameter에 민감한 편임) 만약 Adaboost보다 더 복잡한 boosting 모델을 사용한다면 hyperparemeter를 최적화하는 이슈를 해결하는 것도 논문 주제로 잡을 수 있지 않을까





 
