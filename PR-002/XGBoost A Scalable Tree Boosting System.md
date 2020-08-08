# XGBoost: A Scalable Tree Boosting System



**Abstract**

* 부스팅 트리(Boosting Tree) 는 매우 효과적이고 널리 사용되는 머신러닝 방법이다.
* 이 논문에서 우리는 XGBoost 라는 확장 가능한 (Scalable) End-to-End 트리 부스팅 시스템을 설명한다.
* 우리는 Sparse 한 데이터에 대한 새로운 **희소성 인식 알고리즘 (Sparsity-Aware Algorithm)** 과 **근사적인 트리학습(Approximate Tree Learning)**을 위한 **Weighted Quantile Sketch** 를 제시한다.
* 더 중요한 것은, 우리가 Scalable 한 트리 부스팅 시스템을 구축하기 위해 캐시(Cache) 액세스 패턴, 데이터 Compression 및 Sharding 에 대한 insight 를 제공한다.
* 이러한 통찰을 결합함으로써 , XGBoost 는 기존 시스템보다 훨씬 더 적은 자원을 사용하여, 수십억개 이상의 예제로 Scale 함



**1. Introduction**

* 머신 러닝과 데이터기반 접근법들은 많은 영역에서 매우 중요해졌다
* 스팸 분류기, 광고 시스템, 이상현상 탐지 시스템등 성공적인 어플리케이션으로 이끈 중요한 요인은 `2가지` 이다
  * 복잡한 `데이터 종속성`을 포착하는 효율적인 모델 사용
  * 대규모 Dataset에서 `관심 모델`을 학습하는 Scalable 한 학습시스템

* 실제로 사용되는 머신 러닝 방법중, 그래디언트 부스팅 트리(GBT)는 많은 어플리케이션에서 빛을 발하는 효과적인 기법중 하나

Ex)

* 오픈 소스 패키지이기 때문에 자유롭게 사용 가능함
* Kaggle에서 주최하는 대회에서 대부분의 Winning Solution 이 XGBoost를 사용했으며, Top-10 내 Winning 팀은 모두 XGBoost를 사용했음
* 심지어, 우승팀이 구성한 앙상블 방법은 잘 학습된 XGBoost 보다 약간더 좋은 성능을 발휘한다는 것이 기록됨
* 이러한 결과들을 통해 XGBoost가 광범위한 문제들에서 최신 결과를 제공한다는 것이 증명됨

**XGBoost의 가장 중요한 성공 요인은 모든 시나리오에서 Scalable하다는 것이다**

* 단일 머신을 이용하여 기존 솔루션보다 10배 이상 더 빠르게 실행되며, 분산되거나 제한된 메모리 상황에서도 수십억개 이상의 예제들로 Scalable 한다.
* XGBoost의 Scalability는 일부 중요한 시스템과 알고리즘적인 최적화 덕분
* 이러한 혁신은 다음과 같은 내용을 포함한다.
  * 새로운 트리 학습 알고리즘은 Sparse 한 데이터를 다루기 위한것임
  * 이론적으로 입증된 Weighted Quantile Sketch 절차는 근사 트리 학습에서 인스턴스 가중치를 다룰 수 있게 만들어줌.
  * 병렬 및 분산 컴퓨팅은 학습 속도를 향상시켜 모델 탐색을 가속화한다.
* 중요한것은 XGBoost가 Out-of-Core 계산을 사용하고, 데이터 과학자들이 데스크탑에서 수억개의 예시를 처리할수 있도록 만들어준다
* 마지막으로 이러한 기법들을 결합하고, 최소한의 클러스터 자원을 사용하여 더 큰 데이터로 Scalable 하는 End-to-End 시스템을 만드는것은 흥미로운 일이다



**기존에도 일부 병렬적 부스팅 트리에 대한 연구들이 있었지만, Out-of-Core 계산, 캐시 & Sparsity 인식 학습 방향은 연구되지 않았다.**

* 더 중요한 것은, 이러한 모든 측면들을 결합한 End-to-End 시스템은 실제 사용 사례들에 대해 새로운 솔루션을 제공한다.
* 이는 데이터 과학자 뿐만 아니라 연구원이 강력한 부스팅 트리 알고리즘을 응용하여 구축할수 있게 해준다.
* 이러한 주요 공헌 이외에도, Regularized 학습 목적함수를 제시하고, 완성도를 위해 포함할 Column Sub-Sampling 을 지원하여, 추가적으로 개선함



**2. Tree Boosting in a Nutshell**

**2-1. Regularized Learning Objective**

For a given data set with n examples and m features D = {(xi, yi)} (|D| = n, xi ∈ R m, yi ∈ R), a tree ensemble model (shown in Fig. 1) uses K additive functions to predict the output.

> n 개의 예제와 m 개의 특징이 있는 주어진 세트에 대해 D =  {(xi, yi)} (|D| = n, xi ∈ R m, yi ∈ R), 트리 앙상블 모델 (밑에 그림)은 output을 예측하기 위해 K개의 Additive 함수를 사용한다.

![image-20200808173321507](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808173321507.png)

![image-20200808173437120](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808173437120.png)

where F = {f(x) = wq(x)}(q : R m → T, w ∈ R T ) is the space of regression trees (also known as CART). Here q represents the structure of each tree that maps an example to the corresponding leaf index. 

> 회귀 나무의 공간 (CART)
>
> F = {f(x) = wq(x)}(q : R m → T, w ∈ R T )  
>
> q : 해당하는 잎 인덱스에 예제를 매핑하는 각 트리 구조
>
> T : 트리 잎 갯수
>
> f(k) : 독립적인 나무 구조 q & 잎 가중치 w

* 의사 결정 나무 (Decision Tree) 와 달리, 각 회귀 나무는 잎에 연속적인 점수를 가지고 있으며, 우리는 i번째 잎의 점수를 나타내기 위해 wi를 사용한다.
* 주어진 예시에서, 우리는 트리에서 잎을 분류하기 위해 트리 (q)의 결정 규칙 (Decision Rule)을 사용하고, 해당되는 잎(w)에서 점수를 합산하여 최종 예측을 수행한다.
* 모델에서 사용된 함수 집합을 학습하기 위해서, 다음과 같은 (regularized)목적함수를 최소화 해야한다.

![image-20200808181411597](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808181411597.png)

**수식**

l : 예측값 (yhat(i)), 타겟값(y(i)) 사이의 차이를 측정하는 미분 가능한 Convex 손실함수

Omega : 모델 복잡도에 패널티를 부과 (회귀 나무 함수들)

* Regularization Term 은 over-fitting 을 피하기 위해 최종 학습된 가중치들을 smooth 하게 만듬(정규화)



**2-2. Gradient Tree Boosting**

* 수식(2) 의 트리 앙상블 모델은 함수들을 파라미터로 가지며, 유클리드 공간에서 전통적인 최적화 방법들을 사용하여 최적화 할수 없음
* 대신, 이 모델은 추가적인 방식으로 훈련됨
* 형식적으로 t 번째 반복에서 i 인스턴스 예측값이 (yhat(i)^t(i)) 라고 한다
* 우리는 다음과 같은 목적함수를 최소화 하기 위해 f(i)를 추가해야한다.

![image-20200808182505630](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808182505630.png)

* 일반적인 환경으로 목적함수를 빠르게 최적화 하기 위해 , Second Order Taylor Expansion을 사용하여 근사적으로 계산함

![image-20200808182518878](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808182518878.png)

**수식**

* g(i)