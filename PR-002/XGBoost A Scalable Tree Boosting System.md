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
* 형식적으로 t 번째 반복에서 i 인스턴스 예측값이 y (t) 라고 한다
* 우리는 다음과 같은 목적함수를 최소화 하기 위해  ft 를 추가해야한다.

![image-20200808182505630](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808182505630.png)

* 일반적인 환경으로 목적함수를 빠르게 최적화 하기 위해 , Second Order Taylor Expansion을 사용하여 근사적으로 계산함

![image-20200808182518878](C:\Users\s_m04\AppData\Roaming\Typora\typora-user-images\image-20200808182518878.png)

> 수식
>
> gi = ∂yˆ (t−1) l(yi, yˆ (t−1)) : 손실 함수의 1차( First Order) Gradient 통계량
>
> hi = ∂ 2 yˆ (t−1) l(yi, yˆ (t−1)) : 손실 함수의 2차(Second Order) Gradient 통계량



* 우리는 상수 Terms 제거해서 다음과 같이 단계 (t) 에서 단순화된 목적함수를 얻을수 있었다

![image-20200809151059327](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809151059327.png)

* 잎 (j)의 인스턴스 집합을 로 정의하겠다
* 우리는 Regularization Term (Omega)를 확장하여 , 수식 3(바로 위) 을 다음과 같이 다시 작성할수 있었다.

![image-20200809151202982](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809151202982.png)

* 고정된 구조 (Structure) (q) 의 경우, 다음과 같이 잎(j) 의 최적의 가중치 w ∗ j를 계산할 수 있다.

![image-20200809151301428](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809151301428.png)

* 그리고 다음과 같이 해당하는 최적의 목적함수를 계산할수 있다

![image-20200809151337814](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809151337814.png)

* 수식 6 트리 구조 (q) 의 품질을 측정하기 위해 함수에 점수를 매기는 역할로 사용될수 있음
* 이 점수는 보다 광범위한 목점 함수를 위해 도출된다는 점을 제외하면, 의사결정 나무를 평가하기 위한 **불순도 점수와 같다**
* Figure 2 는 이 점수가 어떻게 계산되는지 설명해준다.

![image-20200809151558557](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809151558557.png)

* 수식 (6) 과 같이 해당하는 최적의 목표 함수 값을 계산할수 있음
* 보통, **가능한 모든 트리구조를 나열하는 것은 불가능하다**
* 대신, 한 잎에서 시작하여 반복적으로 트리를 가지에 추가하는 탐욕 알고리즘이 사용된다.
* IL and IR 이 분할 후의 왼쪽 & 오른쪽 인스턴스 집합이라고 가정하겠음.
* \I = IL ∪ IR, 로 놓은 다음, 분할 후의 손실 감소는 다음과 같이 주어짐 :

![image-20200809151745102](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809151745102.png)

* 이식은 보통 분할 후보들을 추정하는데 사용된다.



**2-3. Shrinkage and Column Subsampling**

* Section 2.1 에서 언급한 Regularized 목적함수 외에도, 추가적으로 **Over-fitting 을 예방하는 두가지 기법**들이 사용된다.

**(1) Shrinkage**

* 부스팅 트리의 각 단계 이후 마다 **새롭게 추가된 가중치를 요인  (eta) 로 Scaling 한다.**
* Stochastic 최적화의 Learning Rate 와 유사하게, **각 개별 트리의 영향도를 감소시키고**, 모델을 향상시키기 위해 **미래 트리 공간을 남겨놓음**

**(2) Column (Feature) Subsampling**

* 이 기법은 Random Forest에서 흔하게 사용되지만, 이전 부스팅 트리에서는 적용되지 않았음
* 유저 피드백에 따르면, 전통적인 Row Subsampling 보다 Over-fitting 을 더 잘 예방해 준다고한다.
* 이 기법을 사용하면 **병렬 알고리즘 계산 속도를 가속화할수있다**



**3. Split Finding Algorithms**

**3-1. Basic Exact Greedy Algorithm**

트리 학습의 핵심적인 문제중 하나는 수식(7) 이 나타내는 것처럼, **가장 좋은 분할을 찾는것이다.**

* 그러기 위해서, 분할 탐색 알고리즘은 **모든 Features 에서 가능한 모든 분할들을 열거한다.**
* 우리는 그것을 exact greedy algorithm 이라고 부른다.
* 기존 단일 머신 부스팅 트리의 대부분에서는 이 알고리즘을 지원하며, Algorithm 1에서 설명한다.

![image-20200809153235178](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809153235178.png)

* **연속적인 Feature에 대해** 가능한 모든 분할들을 열거해야한다.
* 효율적으로 수행하기 위해,**Feature 값에 따라 데이터를 정렬해야 하고**, 수식(7)의 **구조 점수에 대한 Gradient 통계량을 축적하기 위해 정렬된 순서에 따라 데이터를 찾아간다.**



**3-2. Approximate Algorithm**

* EGA는 **가능한 모든 분할 지점들을 탐욕적으로 열거하기 때문에 굉장히 강력함**
* 하지만 메모리에 **데이터가 완전히 적합되지 못할때**, 효율적으로 수행하는 것은 **불가능**함
* 이와 동일한 문제는 **분산된 환경**에서도 발생하게 된다.
* 이 두가지 상황에서 효율적인 GBT를 지원하기 위해서, **근사 알고리즘(AA, Approximate Algorithm)을 사용해야 한다.**
* 부스팅 트리의 AA 프레임 워크는 Algorithm2 에 주어짐

![image-20200809154032226](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809154032226.png)

* 이 알고리즘은 먼저 Feature 분포의 Percentiles에 따라 후보 분할지점을 제시함
* 그런 다음, 연속적인 Feature를 이러한 후보 지점으로 나뉜 Bucket에 매핑하고 통계량을 집계한 다음, 집계된 통계량을 바탕으로 만들어진 Proposal 중 가장 좋은 솔루션을 찾는다.

**Proposal 이 주어진 시기에 따라, 두가지 다른 버전으로 알고리즘이 나뉜다.**

* Global Variant :
  * 초기 트리 구성 단계에서 모든 후보 분할들을 제시하고, 모든 Levels에서 분할 찾기를 수행할 때 동일한 Proposals를 사용한다.
* Local Variant :
  * 매번 분할이 진행된 후에 다시 제시한다.
* Proposal 단계 수 : Global < Local
* 후보 분할 지점 갯수 : Global > Local
  * 매번 분할이 진행된 후에 후보 분할 지점이 개선 (Refine) 되지 않기 때문임
  * Local Proposal은 분할이 진행된 후에 후보를 개선하고, 잠재적으로 더 깊은 트리에 더 적합할수 있음
* Figure 3을 통해 다른 알고리즘과의 비교를 보여준다.

![image-20200809154943226](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809154943226.png)

**3-3. Weighted Quantile Sketch**

**AA의 중요한 단계중 하나는 후보 분할 지점을 제시하는 것이다.**

* 보통 Feature Percentiles 는 후보들이 데이터에 균등하게 분산되도록 하기 위해 사용된다.
* K 번째 Feature 값과 각 훈련 인스턴스의 2차 Gradient 통계량을 나타내는 다중 집합 = {(x1k, h1),(x2k, h2)· · ·(xnk, hn)}이 있다고 한다면
* 우리는 랭크 함수 rk : R → [0, +∞) 을 다음과 같이 정의할수 있다.

![image-20200809160144773](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809160144773.png)

* 이는 Feature 값 k 가 z 보다 작은 인스턴스 비율을 나타낸다.
* 목표는 후보 분할 지점 s {sk1, sk2, · · · skl}, 을 찾는 것이다.

![image-20200809160310629](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809160310629.png)

* (E,epsilon) : 근사치 인자 ( Appoximation Factor )
* 직관적으로, 이는 대략 1/ E 의 후보 지점이 있다는 것을 의미한다.
* 각 데이터 지점은 h(i) 에 의해 가중치됨
* h(i) 가 왜 가중치를 나타내는지 보기 위해, 다음과 같이 수식(3)을 다시 작성할수 있다.

![image-20200809160523146](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809160523146.png)

* 이는 레이벌 g(i) / h(i) 과 가중치 h(i) 를 이용한 정확히 가중치가 적용된 제곱 손실을 의미한다.
* 대규모 데이터셋에서, 이 기준을 만족하는 후보 분할을 찾는 것은 쉽지 않음
* **모든 인스턴스가 동일한 가중치를 가질때,** **Quantile Sketch 이라 부르는 기존 알고리즘은 이 문제를 해결할수 있음**
* 그러나, **가중치가 적용된 데이터셋**에 대한 **Quantile Sketch 는 존재하지 않는다.**
* 따라서, **기존에 존재하는 부스팅 트리 알고리즘 대부분은 실패할 확률이 있거나** **이론적 보장이 없는 데이터의 무작위 Subset에 의존한다**.



**이러한 문제를 해결하기 위해, Distributed Weighted Qunatile Sketch 알고리즘을 제시함**

* 이 알고리즘은 provable,theoretical,guarantee 를 가지고 가중치가 적용된 데이터를 다룰수 있다.
* 일반적인 아이디어는 각 작업에서 일정한 정확도 수준을 유지할수 있도록 보장된 상태에서 Merge and Prune 작업을 지원하는 데이터 구조를 제시한다.



**3-4. Sparsity-aware Split Finding**

* 실제 많은 문제들에서는 Input mathbf(x) 가 Sparese 한 것은 흔한 일이다.
* Sparsity를 유발하는 여러 가지 가능성이 있음:
  * 데이터 내 결측치의 존재 유무
  * 통계량에서 빈번한 0 엔트리
  * One-hot Encoding 같은 Feture Engineering Artifacts

**데이터 내 Sparsity 패턴을 인식하는 알고리즘을 만드는 것은 중요한 일이다**

* 그러기 위해, 우리는 Figure 4 처럼 각 트리 노드 마다 Default Direction을 추가하는 것을 제시함
* Sparse 행렬 mathbf(x) 에서 값이 누락 되었을 경우, 인스턴스는 Default Direction으로 분류를 수행함

![image-20200809162148558](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809162148558.png)

* 최적의 Default Direction 은 데이터에서 학습되며, Algorithm 3 에서 설명한다.

![image-20200809162230815](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809162230815.png)

* 핵심적으로 개선한 것은 결측되지 않은 엔트리 lk 만 방문하는 것
* 제시된 알고리즘에서는 존재하지 않는 값을 결측치로 처리하고, 결측치를 다루는 가장 좋은 Direction을 학습함
* 우리가 알고 있는 한도 내에서, 기존 트리 학습 알고리즘 대부분은 Dense한 데이터에만 최적화 되거나, 범주형 (Categorical) Encoding 같은 제한된 경우들을 다루는 구체적인 절차들이 필요함
* XGBoost 는 모든 Sparsity 패턴을 통합된 방식으로 다룬다.
* Figure 5 는 Sparsity 인식 알고리즘과 Naive 한 구현 알고리즘을 비교함

![image-20200809162432639](../../../../AppData/Roaming/Typora/typora-user-images/image-20200809162432639.png)