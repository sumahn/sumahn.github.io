---
title: Induction of Decision Trees
excerpt: Review

categories:
 - Statistics
tags:
 - Machine Learning
last_modified_at: 2022-01-03T15:39:00-22:00
use_math: true
---



*Induction of Descision Trees - J.R.Quinlan*



### Introduction 

1950년대 중반부터 AI가 학문으로 자리잡으면서 머신러닝 분야가 대두되었다. 학습 능력은 지적 행동의 특징이고, 따라서 지능을 어떠한 현상으로 이해한다는 것은 학습에 대한 이해를 포함해야 한다. 더 구체적으로 표현하면 학습은 고성능 체계를 만드는 기반이다. 

학습에 대한 연구분야는 여러 분야로 이루어져 있는데, 한 편에서는 스스로의 성능을 모니터링하고 모수(parameter)를 조정함으로써 성능을 스스로 개선하려는 adaptive system이 사용되고 게임이나 문제 해결 등 다양한 분야에 적용된다. 다른 한 편에서는 학습을 개념의 형태로 구조화된 지식을 얻는 것으로 간주한다. 후자와 같은 종류의 머신러닝의 실질적 중요성은 knowledge-based expert systems의 출현에 의해 강조되었다. 이 시스템은 명시적으로 드러난 지식에 의해 동작하고 이 지식은 도메인 전문가와 knowledge engineer 사이의 상호작용을 통해 구축되어 왔다. 

문제가 단순하다면 1인당 하루에 몇 개의 rule만 가지고도 해결을 할 수 있지만, 문제가 복잡하다면 수백 수천 개의 rule이 요구될 수 있다. 인터뷰 같은 방식으로 지식을 얻는 것은 이 접근을 따라갈 수 없고, 이 현상을 "병목 현상(bottleneck)"이라고 한다. 이런 인식이 지식을 설명하는 방법으로서 머신러닝의 발전을 촉진시켰다. 

이 논문에서는 머신러닝의 극히 일부분을 다루고, 간단한 종류의 knowledge-based system을 구축하기 위해 사용되어온 학습 체계에 대해서 다룬다. 

<br/>

### The TDIDT(Top-Down Induction of Decision Trees) family of learning systems 

Carbonell, Michalski, Mitchell (1983)은 머신러닝 시스템을 분류할 세 가지 주요 차원을 정립했는데, 

1. The underlying learning strategies used 
2. The representation of knowledge acquired by the system 
3. The application domain of the system 

이다.   



#### 3. The application domain of the system 

이 논문에서는 이 세 가지 차원과 밀접한 연관을 가진 학습 체계에 대해 다룬다. 시스템의 적용 분야부터 생각해보면 이 시스템들의 적용 가능 분야는 어떤 특정 분야로 한정되지 않는다. 또한 일반적 목적으로 사용되지만 시스템이 다루고자 하는 적용은 모두 **분류(classification)**를 포함한다. 

>  **The product of learning is a piece of procedural knowledge that can assign a hitherto-unseen object to one of a specifed number of disjoint classes.**

이 말이 무슨 말인가하면 내가 여러 마리의 개와 고양이를 학습했다고 한다면, 지금까지 보지 못했던 새로운 개와 고양이를 보더라도 개인지 고양이인지 알 수 있는 것. 그것이 학습의 산물이라는 것이다.  

분류(classification)가 procedural task의 아주 조그마한 부분으로 보일 수 있지만 robot planning과 같은 활동들도 분류 문제로 치환할 수 있다. 따라서 분류라는 작업은 굉장히 의미있는 일!  

#### 2. The representation of acquired knowledge 

아무튼 넘어가서 2번째 TDIDT family에 속한 알고리즘은 획득한 지식을 의사결정나무(decision tree)로 표현하는 특징이 뚜렷하다. 의사결정나무는 상대적으로 단순한 knowledge formalism이기 때문에 TDIDT family에서 사용되는 학습 방법론 또한 단순하다. 그럼에도 복잡한 문제를 푸는 것이 가능하다는 장점이 있다.   



#### 1. The underlying strategy 

> **The underlying strategy is non-incremental learning from examples.**

  

학습 체계들은 분류 학습과 관련된 경우의 집합으로 주어지고 의사결정나무를 위에서부터 아래로, 샘플에서 주어지는 frequency information을 따라 발전시킨다. 하지만 여기서 어떤 샘플의 특정한 순서를 고려하지는 않는다. 이것은 MARVIN(Sammut, 1985)에서 사용되는 incremental methods와 상반된다. (이 학습 체계에서는 샘플이 주어지는 순서가 중요하다.) 이 논문에서 다루는 학습 체계는 주어진 샘플의 패턴을 학습 동안 계속해서 반복해서 조사하는 것이 가능해야 한다. 이러한 데이터 기반 접근을 사용하는 다른 알고리즘으로는 BACON(Langley, Bradshaw and Simon, 1983)이나 INDUCE(Michalski, 1980) 등이 있다. 

<br/>

**사실 위의 말들 다 이해할 필요는 없고 이 논문에서 다루는 것은 의사결정나무라는 것이고, 이 나무들이 어떻게 구성되어서 뻗어나가는지를 이해하면 된다.**  



<img src="/assets/images/TDIDT_family_tree.jpeg" width="400" height="400"/>

  

**CLS(Concept Learning System; Hunt, Marin and Stone, 1966)**는 분류 cost를 최소화하려고 하는 의사결정나무를 구성한다. 여기서의 cost는 두 가지로 나뉘는데,

1. 어떤 물체가 나타내는 속성 A의 값에 대한 측정 오류 
2. 오분류

각 단계마다 CLS는 fixed depth인 가능한 의사결정나무 공간을 탐색하고, 이 제한된 공간 상에서 cost를 최소화하는 방향을 선택한 뒤 트리를 뻗어나간다. depth를 얼마로 고정시키느냐에 따라서 매우 많은 계산량을 요구할 수도 있지만 계산량을 늘리면 애매한 패턴을 찾아낼 수도 있기 때문에, 적절하게 조절하는 것이 필요하다. 

  

**ID3(Quinlan, 1979, 1983a)**는 체스 게임에서 특정 위치에서 몇 번의 고정된 움직임으로 승리할 수 있는지를 결정하는 induction task에 대한 반응으로 CLS를 발전시킨 알고리즘 중 하나이다. ID3는 DT에서 중요한 편에 속하기 때문에 후에 더 다루기로 한다. 

  

**ACLS(Paterson and Niblett, 1983)**은 ID3의 일반화 버전이다. CLS나 ID3는 어떤 객체를 묘사하는 특성이 특정 집합에서의 한정된 값을 취해야 하는 조건이 있는데, ACLS는 unrestriced integer value를 갖는 속성도 허용한다. 이런 특성 때문에 ACLS는 이미지 인식 문제 같은 분야에서도 사용될 수 있다. 

  

**ASSISTANT(Kononenko, Bratko and Roskar, 1984)**는 ACLS의 unrestricted integer value를 넘어서 continuous real values를 갖는 속성도 허용한다. 이 알고리즘은 ID3처럼 iterative하게 의사결정나무를 학습시키지 않고, 사용가능한 objects로부터 "좋은" training set을 선택하는 알고리즘을 포함하나. 

  

맨 아래의 세 개는 ACLS를 사용자 친화적으로 업그레이드한 버전이다. 

<br/>

### The induction task

아래와 같은 training set이 있다고 가정하자.

<img src="/assets/images/Table1.jpeg" height="500" align="center">

 induction task는 속성값으로 object의 class를 분류하는 rule을 세우는 것이다. 어떻게 classification rule을 세울 것인가? 의사결정나무가 그 역할을 할 수 있는데, 가장 단순하게는 아래 그림과 같이 분류할 수 있다. 

<img src="/assets/images/Figure2.jpeg" height="400" align="center">

위에서부터 아래로 해석하면 되는데, 먼저 주어진 training set에서 outlook이 overcast인 경우는 모두 class가 P이므로 P로 결론짓는다. outlook이 sunny인 경우는 humidity 속성이 high인지 normal인지에 따라 class를 구분지을 수 있고, outlook이 rain인 경우에는 windy 속성이 true인지 false인지에 따라 class를 구분지을 수 있다. 

만약 모든 속성이 adequate(_속성이 같으면서 다른 class에 속하는 경우가 없는 경우_)하다면, 항상 의사결정나무로 분류를 정확하게 할 수 있다. induction의 필요성은 training set만이 아니라 새롭게 관측되는 값에서도 분류를 잘하는 의사결정나무를 구성하는 데에 있다. 이렇게 하기 위해서는 **한 객체의 class와 속성값의 의미있는 관계성을 발견해야만 한다.**

<br/>

### ID3 

우리가 아직 관측하지 못한 데이터에 대해서도 예측을 잘 하기 위해서는 training set에 과적합되지 않고 단순한 모형이 더 좋다.(_Tradeoff of Bias and Variance_) 그렇다면 한 가지 시도해볼 수 있는 것은 training set을 정확하게 분류해낼 수 있는 의사결정나무 중에서 가장 단순한 모형을 선택하는 것이다. 하지만 이런 나무의 개수는 너무 많기 때문에 computation 측면에서 굉장히 비효율적이기 때문에 small induction task에 대해서 적절하다. 

**ID3**는 많은 attribute(feature)와 objects가 있지만 computation이 많지 않은 tree가 상대적으로 더 좋은 측면에서 설계되었다. 일단 ID의 기본 구조는 iterative하다. Window(training set의 부분집합)이 랜덤하게 선택되고 의사결정나무도 이 window에 맞게 생성된다. 그 후에 뽑히지 않은 나머지 set으로 테스트한다. 만약에 window랑 window에 뽑히지 않은 set에 대해서 정확도가 높다면 전체 training set을 정확하게 분류한다고 판단할 수 있고, 프로세스를 종료하면 된다. 잘 분류하지 못하면 오분류한 sample을 window에 포함시킨 후에 프로세스를 이어간다. 이렇게 하면 몇 번의 iteration으로 정확도를 높인 의사결정나무를 만들 수 있다고 한다. O'Keefe는 window에 전체 training set이 포함되지 않으면 iterative framework가 final tree로 converge한다고 보장할 수 없다고 지적했는데, 이 논문 당시에는 아직 이런 일은 없었다고 한다.(지금은 이 문제가 발생하고 이미 해결했을 수도 있다.)

문제의 핵심은 임의의 객체 집합 C에 대해서 어떻게 의사결정나무를 만드는가이다. 만약 $C$가 공집합이거나 하나의 class에 속하는 객체들만을 갖는다면, 의사결정나무는 굉장히 쉽게 구성될 수 있다. 하지만 $T$를 $O_1, O_2, \cdots, O_w$를 결과로 가질 수 있는 test set이라고 하면, $C$에 속한 각각의 객체는 $T$의 possibile outcome 중 하나를 갖기 때문에 $C$는 $\{C_1, C_2, \cdots, C_w\}$로 나타낼 수 있다. 그리고 $C_i$ 는 $O_i$를 결과로 갖는 객체들을 포함한다. 

만약 $C_i$ 각각이 $C_i$에 대한 의사결정나무로 대체될 수 있다면 모든 $C$에 대한 의사결정나무를 만들 수 있다는 것은 직관적으로 이해할 수 있다. 잘못하면 각각의 subset이 하나의 class로 구성된 집합을 만들어버릴 수 있기 때문에 간단한 의사결정나무를 만들 때는 test set을 선택하는 것은 매우 중요하다. 

ID3는 두 개의 가정을 베이스로 하는 information-based method를 사용한다. $C$가 class $P$의 $p$ object와 class $N$의 $n$개의 object를 포함한다고 하자. 이 때의 가정은 

1. Any correct decision tree for $C$ will classify objects in the same proportion as their representation in $C$. An arbitrary object will be determined to belong to class $P$ with probability $\frac{p}{(p+n)}$ and to class $N$ with probability $\frac{n}{(p+n)}$
2. When a decision tree is used to classify an object, it returns a class. A decision tree can thus be regarded as a source of a message $'P'$ or $'N'$, with the expected information needed to generate this message given by  $I(p, n) = -\frac{p}{(p+n)}log_{2}\frac{p}{(p+n)} - \frac{n}{(p+n)}log_{2}\frac{n}{(p+n)}$

  

첫 번째 가정은 각각의 객체가 어떤 클래스에 속할 확률이 동일하다는 것이고, 두 번째 가정은 의사결정나무는 class가 P인지 N인지 결정하는 역할을 하고, 기대 정보량은 위와 같이 주어진다는 것이다. 

만약 attribute $A$가 $\{A_1, A_2, \cdots, A_v\}$를 값으로 갖고, 의사결정나무의 root에서 사용된다면 $C$를 $\{C_1, C_2, \cdots, C_v\}$로 나눌 것이다. 이때 $C_i$가 class $P$에 속하는 $p_i$ 개의 object와 class $N$에 속하는 $n_i$개의 object를 갖는다고 하면, $C_i$에 대한 subtree로 요구되는 expected information은 $I(p_i, n_i)$이다. 그럼 $A$를 root로 삼는 tree에 요구되는 expected information은 다음과 같이 가중평균을 통해 구할 수 있다.   

$E(A) = \sum_{i=1}^{v}\frac{p_i+n_i}{p+n}I(p_i, n_i)$

따라서 A에서 가지를 침으로써 얻을 수 있는 정보량은 $gain(A) = I(p, n) - E(A)$이다. 

**ID3의 장점은 class와 attribute간의 관계를 쉽게 잘 보여준다는 점이다.** 그러면서도 꽤나 괜찮은 predictive accuracy를 확보한다는 점인데, training set과 test set을 분리하고 test set에 대해서 예측력을 실험한 결과 준수한 성능을 보였다고 한다. 

마지막으로 computational complexity는 어떨까? 의사결정나무에서 각각의 노드에서 속성 A가 모두 탐색됨과 동시에 각각의 class $C$개를 모두 탐색해야 하기 때문에 각 노드에서의 time complexity 는 $O(|C| * |A|)$이다. 여기서 $|\cdot|$은 개수를 의미한다. 따라서 ID3의 total computational complexity는 속성의 개수와 training set의 사이즈에 비례한다고 할 수 있다.

<br/>

### Noise

지금까지 다룬 얘기는 training set이 정확한 정보를 제공한다는 전제가 깔려 있다. 하지만 실제로 training set은 정확하지 않을 수 있다. Training set 자체도 주관적인 요소의 개입 혹은 측정 오류 등으로 인해 이미 오분류되어 있을 수 있다. 이런 상태로 의사결정나무를 학습시킨다면 어떤 attribute를 inadequate한 것으로 취급해버릴 수 있고, 잘못된 complexity를 갖는 의사결정나무를 만들 수 있다. 이런 attribute 값에서나 class의 non-systemic error를 **noise**라고 한다. noise를 다룰 수 있는 모형을 만들기 위해서는 두 가지 개선점이 요구된다. 

1. Inadequate한 attribute를 처리할 수 있어야 한다. 
2. attribute를 추가해도 예측 정확도가 개선될지 안될지를 결정해야 한다. (노이즈 하나 때문에 complexity가 증가하는 것을 막기 위해서)



2번째 문제를 해결하는 방법 중 하나는 information gain에 threshold를 줘서 threshold보다 낮으면 tree를 뻗어나가지 않는 것이다. 하지만 이 threshold를 어떻게 정할 것인가에 대한 문제가 남기 때문에 그리 좋은 해결책은 아니다. 더 좋은 해결책으로 카이제곱 검정을 통해서 class와 attribute간의 dependency를 알아내는 것이다. 이 과정을 통해서 irrelevance가 높은 신뢰수준에서 기각되지 않는다면 tree를 뻗어나가지 않는 형식으로 노이즈에 과적합돼서 complexity를 높이는 것을 막을 수 있다. 

1번째 조건으로 발생할 수 있는 문제는 collection $C$가 두 class를 대표하는 점을 포함함에도 불구하고 attribute가 inadequate하거나 class와 irrelevant하다는 이유로 testing에서 배제될 수 있다. 따라서 이 경우에는 class 정보를 담는 leaf를 만들 필요가 있다. 그 후 $\frac{p}{(p+n)}$을 계산해서 'class P에 속할 확률'로 해석하는 접근, 혹은 $p > n$이라면 class $P$로 반대의 경우라면 $N$으로 결정하는 접근을 해볼 수 있다. 첫 번째 접근은 $C$의 object에 대한 오차제곱합을 줄이고, 두 번째 접근은 $C$에 대한 절대오차합을 줄인다. 평균 오차를 줄이려면 두 번째 접근이 더 좋다고 하는데 결국엔 case by case이다. 

<br/>

### Unknown attribute values

이전 파트에서는 속성값이 잘못되는 등의 noise일 때를 다뤘다. 속성값이 결측치라면 어떻게 해야할까? 주어지는 정보를 참고하여 채워넣을 수도 있을 것이다. **ASSISTANT** 알고리즘은 Bayesian 방법을 통해서 class $C$에서 $A$의 분포를 통해서 object가 attribute $A$에서 $A_i$를 가질 확률을 구해 imputation한다. object가 attribute $A$에서 $A_i$ 값을 가질 확률은 다음과 같이 표현된다. 

$$Pr(A = A_i | class =P) = \frac{Pr(A = A_i , class = P)}{Pr(class = P)} = \frac{p_i}{p}$$

이렇게 구한 뒤에 Likelihood를 최대화하는 값으로 채워넣으면 된다. 

Alen Shapiro는 의사결정나무를 통해서 imputation하는 것을 제안했다. attribute $A$가 정의된 set에 대해서 class 또한 하나의 attribute로 취급한 뒤 $A$에 대한 의사결정나무를 만든 뒤 정의되지 않은 속성 $A$가 예측해야 되는 test class로 취급하는 것이다. 

아이디어는 되게 좋아보이는데 unknown attribute가 1개일 때는 성능이 엄청 안좋다고 한다. 논문에서는 Bayesian, DT, 단순히 가장 많이 나오는 값으로 채우는 방법을 비교했는데 결과는 다음과 같다. 

<img src="/assets/images/Table3.jpeg" height="300" width=700 align="center">

Bayesian 방법은 단순 imputation 방법보다 살짝 좋고, DT가 생각보다 괜찮은 결과를 보였다. 

이렇게 채우지 말고 DT는 또다른 방법을 적용해볼 수 있는데 바로 결측치를 하나의 category로 간주하는 것이다. 그런데 이렇게 되면 결측치가 있는 것이 없는 것보다 information gain이 더 높다고 판단해버릴 가능성이 있다. 이건 우리가 원하는 결과가 아니기 때문에 다른 접근을 생각해볼 필요가 있다. 

attribute $A$가 $\{A_1, A_2, \cdots, A_v\}$를 갖는다고 하자. object의 집합 $C$에 대해서 $A_i$를 갖는 object의 개수를 $p_i, n_i$라고 하고 모르는 값들을 갖는 object의 개수를 $p_u, n_u$라고 하자. attribute $A$의 infromation gain이 평가될 때, 모르는 값을 갖는 object는 $C$에서 상대적 빈도에 비례해서 A의 값들로 분배된다. 따라서 실제 $p_i$가 마치 $p_i + p_u * \frac{p_i+n_i}{\sum_{i}{(p_i+n_i)}}$인 것처럼 간주하고 평가되기 때문에 unknown value에 대해서는 information gain이 감소하게 된다. 따라서 selection criterion에 의해 attribute가 선택되면 그 attribute에서 unknown value를 갖는 object는 tree를 만들기 전에 버려진다.

<br/>

### The selection criterion

이전에는 의사결정나무의 root를 구성할 때 사용하는 evaluation function으로 information gain을 사용했었다. 그런데 많은 분야에서 information gain이 높다고 나온 attribute는 사실 다른 attribute에 비해 관계가 적었고, Kononenko에 의하면 gain criterion은 다양한 값을 갖는 attribute를 선호하는 경향이 있기 때문이다. 심지어는 attribute A를 random value로 놓고 같은 값이 나올 확률이 낮게끔 다양한 값을 갖게 한다면 random한 값임에도 의사결정나무의 root를 분할하는데 사용한다. 

ASSISTANT가 이 문제를 해결하는데, 모든 가능한 값에 대해서 가지를 치는 것이 아니라 두 개(값에 속하는 것 vs 나머지)로만 쪼개는 구조를 취한다. 이렇게 하면 더 작은 의사결정나무로 더 좋은 성능을 낸다고 한다. 이렇게 binary로 쪼개나가는 형태는 CLS와 유사하지만 single value가 아닌 value의 집합으로 나눈다는 점에서 차이가 있다. 물론 continuous value에 대해서도 이진 분류가 가능하다. 두 개의 중앙값으로 분할 threshold를 잡는다고 생각하면 된다. 

그런데 이런 ASSISTANT의 해결책에는 두 가지 부작용이 따르는데... 첫 번째는 관계없는 attribute끼리 하나로 묶이면서 사람이 해석하기에 어려워졌다는 점이고, 두 번째는 computation 양이 엄청 증가한다는 것이다. value가 $v$ 개가 있다면 이걸로 만들 수 있는 부분집합의 개수는 $2^{v}$ 개이고, 순서 고려를 안하고 공집합을 빼더라도 $2^{v-1}-1$ 개가 만들어진다. $v$가 작으면 문제가 안되는데, $v$가 커질수록 연산량이 굉장히 커진다는 문제점이 있다. 

attribute A의 value가 얼마인가? 라는 것에 대한 information은 다음과 같이 나타낼 수 있다. 

$$IV(A) = - \sum_{i=1}^{v}\frac{p_i + n_i}{p+n}log_{2}\frac{p_i +n_i}{p+n}$$

$gain(A)$는 $A$를 root에서 사용했을 때 정보 요구량의 감소를 말하기 때문에 $gain(A) / IV(A)$ 가 클수록 좋은 선택이 된다. 

그리고 A의 value가 여러 가지일수록 predictive accuracy가 높은 성능을 보였는데, 너무 많은 value를 갖게 되면 training set을 굉장히 작게 쪼개기 때문에 오히려 성능이 떨어지는 결과를 낮는다. 

<br/>

### Conclusion

이 논문은 의사결정나무가 꽤나 robust하면서 빠르고 정확하다는 것을 보인다. 하지만 현재 기준에서는 정확도의 정도가 떨어지기 때문에 여러 기법들을 사용해서 업그레이드시킨다. 하지만 어떤 방식으로 접근하는지, 모형의 단점을 개선하는지에 대한 아이디어 등은 의미가 있다. 





  





