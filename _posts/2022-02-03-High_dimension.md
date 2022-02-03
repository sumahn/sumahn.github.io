---
title: "고차원 데이터(High-dimensional data)"

categories:
 - Statistics

tags:
 - Machine Learning
 - High dimension

last_modified_at: 2022-02-03T18:01:00-05:00
---



## 고차원 데이터(High-dimensional data)

_연세대학교 김일문 교수님의 "Statistics for high dimensional data"를 공부하고 정리한 내용이 포함되어 있습니다._



### 고차원 데이터란 무엇인가? 

고차원 데이터는 관찰값의 개수($n$)보다 속성의 개수($d$)가 큰 데이터를 말한다. 정의 자체는 단순한데 고차원 데이터를 왜 따로 다루는가에 대한 의문이 들 수 있다.

1. 통계학적 관점에서 생각해보면, 중심극한정리(_CLT_)나 대수의 법칙(_Law of large numbers_) 등의 대다수의 정리는 $n \rightarrow \infty$를 가정한다. 고차원 데이터에서는 정통 통계학의 정리들이 성립하지 않기 때문에 새로운 이론의 정립이 필요하다. 
2. 현대에 들어서 고차원 데이터의 비중은 굉장히 증가했다. 이미지 데이터, 의료 데이터, 유전자 데이터 등은 고차원 데이터인 경우가 많기 때문에 이를 분석하기 위해 따로 이론을 정리할 필요가 있다. 



<br/>



### 고차원 데이터일 때 무엇이 문제되는가? 

고차원 데이터에서 정통 통계이론을 적용했을 때 발생할 수 있는 예로 공분산 추정이 있다. (사실 고차원일 때 추정이 가장 문제가 된다. 선형회귀를 생각해보면 $X^TX$의 역행렬이 존재하지 않기 때문에 계수가 identifiable하지 않고, 계수의 분산 또한 커지는 문제가 있다.)





#### 공분산 추정(Covariance Estimation)

예를 들어, $X_{1}, \cdots, X_{n} \in \mathcal{R^{d}}$ 이 평균이 0이고, 분산이 $\Sigma$인 분포를 따른다고 한다면, 


$$
\hat{\Sigma} = \frac{1}{n}\sum_{i=1}^{n}X_{i}X_{i}^T
$$


로 $\Sigma$를 추정해볼 수 있다. 저차원($n > d$)의 경우를 생각하면 대수의 법칙에 따라 $\hat{\Sigma}$는 $\Sigma$의 consistent estimator가 됨을 쉽게 알 수 있다. 이 경우 $ \lvert\lvert \hat{\Sigma} - \Sigma \rvert \rvert_{op} $ 의 분포가 0에 집중되어 있는 형태를 보여야 한다. 하지만 $\frac{d}{n} \rightarrow \alpha \in (0,1)$ 일 경우  가$ \lvert\lvert \hat{\Sigma} - \Sigma \rvert \rvert_{op} $ 가 0에 수렴하지 않는 결과를 보이고 Marchenko-Pastur 분포를 asymptotically 따른다. 



예를 들어서 $\Sigma = I_{d}$라고 하면, $\Sigma$의 eigenvalue의 분포는 1에 집중되어 있는 분포를 보여야 한다. 하지만 실제 분포는 아래와 같다. 





 <p align="center">
   <img src= "/assets/images/High_dim/Enpirical_eigenvalue_covariance.jpeg" width=700>
 </p>



저차원의 경우라면 n이 커짐에 따라 파란색 분포와 같은 분포를 이뤄야 하지만, 고차원의 경우 이것이 성립하지 않음을 확인할 수 있다. 



<br/>



#### 차원의 저주(Curse of Dimensionality)

차원의 저주는 유클리디안 공간(Euclidean space)에서 차원이 늘어남에 따라 계산량은 기하급수적으로 증가하지만 불필요한 noise 또한 급격히 증가하는 것을 말한다. 



아래 그림에서 초록색 원이 필요한 정보를 담고 있다고 가정하면 차원이 2에서 3으로 늘어남에 따라 전체 공간에 비해 불필요한 정보가 있는 공간 또한 증가함을 볼 수 있다. 

<p align="center">
  <img src= "/assets/images/High_dim/curse_dim1.png" width=700>
</p>

이 차원을 계속해서 늘려가다보면 아래 그림과 같이 불필요한 정보만 담게 되는 **차원의 저주**가 발생하게 된다. 

<p align="center">
  <img src="/assets/images/High_dim/curse_dim2.png" width=700>
</p>





<br/>



### 고차원 데이터의 문제를 어떻게 해결할 것인가? 

이 문제를 해결하는 방법은 여러 가지 연구가 되는 중이고, 분야별로 어떻게 적용하는지도 차이가 존재한다. 일반적으로 고차원 데이터를 다루는 방법에는 

1. 차원 축소(Dimension Reduction): 고차원을 저차원으로 줄이는 PCA나 manifold learning 등이 있다. 
2. 고차원에 robust한 모형의 사용

등이 있다. 



이후에 이 문제를 어떻게 해결할 수 있는지를 더 공부해보고자 한다. 





<br/>

### 출처 

[Curse of dimensionality](https://www.researchgate.net/publication/327498046_The_Curse_of_Dimensionality_Inside_Out)

Martin J. Wainwright - High-Dimensional Statistics A Non-Asymptotic Viewpoint 

