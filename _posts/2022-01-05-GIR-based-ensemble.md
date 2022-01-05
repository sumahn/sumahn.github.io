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

### Introduction 

이 논문은 **imbalanced learning**을 다루기 때문에 먼저 **imbalance하다는 것은 무엇인가**에 대해서 먼저 정의할 필요가 있다. imbalance하다는 것은 _the class distribution is uneven for a given imbalanced data set_, 즉 주어진 데이터의 class 분포가 동일하지 않다는 것을 의미한다. 그럼 이제 이게 왜 중요하고 issue가 되는가에 대해 생각해볼 수 있는데, 먼저 생각해볼 점은 기존의 머신러닝 모형은 balanced data set을 생각하고 만들어진 모형이라는 점이다. 이 모형으로 학습을 하게 되면 하나의 많은 class(majority)가 적은 class(minority)를 dominate하는 현상이 발생한다. 이렇게 되면 majority에 대해서 학습이 잘 되기만 하면 모형은 학습이 잘 된 것으로 판단한다. 예를 들어서 majority class: minority class가 90:10이라면 90에 대해서만 잘 분류해도 정확도가 90%라는 좋은 성과를 내게 된다. 하지만 우리는 minority에 관심이 가고 가치있는 경우가 많다. 

두번째로 생각해볼 수 있는 점은 실제로 imbalance한 데이터인 경우가 굉장히 많기 때문이다. 의료 데이터나 사고, 마케팅, 정부 정책 등에 대한 데이터를 보면 minority가 더 값진 경우가 많다. 암에 걸린 환자인지 아닌지 분류하고자 한다고 하면 암이 아닌 환자가 훨씬 많을 것 아닌가. 따라서 현실적으로도 imbalance data에 대해서 어떻게 학습을 할 것인지는 중요한 연구라고 할 수 있다. 

위에 두 가지 측면으로 제시한 점에서 뽑아낼 수 있는 우리의 목표는 결국 **majority에 대한 최소의 cost로 minority를 분류 성능을 높이는 것**으로 세울 수 있다. 

이 문제를 해결하기 위해서 많은 논문들이 나왔는데 그 중에서 SMOTE(Systhetic Minority Over-sampling TEchnique)이나 ADASYN(ADAptive SYNthetic sampling approach), ensemble-based undersampling등이 효과적이고 간단한 기법으로 알려져 있다. 이런 sampling based approach에서 가장 중요한 이슈 중에 하나는 어떻게 새롭게 만들 데이터셋의 가장 좋은 balance를 맞출 것인가이다. 대부분의 approach는 majority와 minority 사이의 class imbalance를 측정하기 위해 sample size ratio를 사용한다. 그렇게 해서 majority와 minority의 샘플 개수를 맞춰 주는데, 이렇게 맞춰줘도 **class imbalance는 아직 남아 있을 수 있다..!** 이 점을 해결하려고 이 논문이 나온건데, 이 논문에서는 class imbalance의 새로운 measure 방법으로 **Generalized Imbalance Ratio(GIR)**을 도입한다. 

이것말고도 생길 수 있는 문제가 하나 더 있다. 일반적으로 oversampling을 하거나 undersampling을 하면 별로 의미 없는 minority class가 생성되거나 의미 있는 majority class가 버려지면서 classfier의 성능이 저하되는 현상이 발생하곤 하는데, 이 논문에서는 불균형 학습 문제를 undersampling이나 oversampling을 써서 여러 개의 balanced subproblem으로 바꾸고, classfier들이 잘 분류하지 못하는 샘플들에 더 집중할 수 있게 만든 다음, 최종적으로 strong classifier를 세우는 ensemble 기법을 사용하면서 해결한다. 그리고 GIR을 써서 undersampling을 하는지 oversampling을 하는지에 따라서 GIREnUS와 GIREnOS로 나눠서 설명한다. 

결국 논문의 키워드를 정리하면 다음과 같이 세 가지로 나타낼 수 있다. 

- GIR, which measures class distribution imbalance between two classes, based on their intra-class coherence
- Adaptively split the imbalanced learning problem into multiple balanced learning subproblems
- GIREnUS, GIREnOS



