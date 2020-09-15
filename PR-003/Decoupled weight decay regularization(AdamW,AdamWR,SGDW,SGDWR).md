# Decoupled weight decay regularization

처음 배울때 Adam 은 "optimizer를 사용할때 Adam을 사용하는 것이 최적의 성능을 보장한다" 라고 배웠었다. 하지만 이번 Dacon 대회를 하면서 adam을 사용하는것보다 AdamW를 사용하는것이 더 좋은 결과를 내는것을 알았고 왜 그런지 논문을 읽어보게 되었다.

AdamW을 소개한 논문 "Decoupled weight decay regularization" 에서는 `L2 regularization`  과 `weight decay` 관점에서 Adam이 SGD 에 비해 일반화 능력이 떨어지는 이유를 설명하고 있다.

> L2 regularization :기존 cost function 에 가중치의 제곱을 포함하여 더함으로 L1 regularization 과 마찬가지로 가중치가 너무 크지 않은 방향으로 학습되게 되는데 이걸 **Weight decay** 라고도 한다.
>
> L2 Regularization 을 사용하는 Regression model 을 Ridge Regression 이라고 부른다.



**시작전 L2 regularization 과 weight decay 정리**

## L2 regularization

L2 regularization 은 손실함수에 weight 에 대한 제곱텀을 추가해줘서 오버피팅을 방지해주는 방법이다. t번째 미니배치에서 손실함수를 f(t), weight를 0 라고 한다면 L2regularization을 포함한 손실함수 f(t)**reg는 다음과 같이 나타낼수 있다.

![image-20200915140527977](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D1.png?raw=true)

λ′는 regularization 상수로 사용자가 설정하는 하이퍼 파라미터이며 다음과 같이 나타낼수도 있다고한다.

![image-20200915140633606](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D2.png?raw=true)

L2 regularization 이 어떻게 오버피팅을 방지를 해줄수 있는지를 많이 찾아보면 수학적인 답변을 찾을수는 없다고 한다.

![image-20200915140818180](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/PRML.png?raw=true)

하지만 직관적인 예시로 사인그래프에서 노이즈를 추가하여 샘플링한 데이터를 polynomial regression 을 사용해 피팅한 그림을 찾았다. M 은 다항식의 최대차수를 의미한다. 사인 그래프를 9차 다항식으로 피팅했을때 다항식의 각 계수는 오른쪽 표의 가장 오른쪽에 나타나있다. 그 값이 굉장히 큰 것을 알수 있다. 굳이 weight를 증가시켜 가면서 local noise에 반응한것으로 볼수 있다고 한다. 

하지만, weight 가 굉장히 클경우 데이터가 아주 조금만 달라져도 예측값이 굉장히 민감하게 바뀌게 된다. 오버피팅된 모델에 훈련 데이터와 다른 테스트 데이터가 들어갔을때 에러가 큰 이유이다.

때문에 손실함수에 L2 regularization을 추가해줌으로써 weight가 비상식적으로 커지는 것을 방지할수 있다. 손실함수를 최소로 만들어주는 것이 목표인데 weight 값이 커지게 되면 손실함수가 커지게 되기 때문이다. 따라서 weight가 너무 커지지 않는 선에서 원래의 손실함수를 최소로 만들어주는 weight를 찾게 되는 것이다.

## Weight Decay

weight decay 는 gradient descent에서 weight 업데이트를 할때. 이전 weight의 크기를 일정 비율 감소시켜줌으로써 오버피팅을 방지한다. 원래 gradient descent 업데이트 식은 다음과 같다.

![image-20200915141356651](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D3.png?raw=true)

알파는 learning rate 이며 여기서 weight decay를 포함하면 다음과 같은 업데이트 식을 사용한다 한다.

![image-20200915141445089](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D4.png?raw=true)

λ 는 decay rate 라고 부르며 사용자가 0과 1사이 값으로 설정하는 하이퍼파라미터이다. weight를 업데이트할때 이전 weight의 크기를 일정 비율만큼 감소시키기 때문에 weight 가 비약적으로 커지는 것을 방지할수 있다.



## L2 Regularization == Weight decay?

위에서 적은것과 같이 두개를 찾아 보았을때 서로 같은것이라고 했다. 하지만 그중에 다르다고 하는 곳이 있었다.

L2 regularization 과 weigth decay 는 일부는 맞고 일부는 틀리다고 한다. 

SGD에서는 맞지만 Adam에서는 틀리다고 한다. Adam을 포함한 adaptive learning rate 를 사용하는 optimizer들은 SGD와 다른 weight 업데이트 식을 사용하기 때문이며 L2 regularizion이 포함된 손실함수를 Adam을 사용하여 최적화 할 경우 일반화 효과를 덜 보게된다고 한다.

**SGD**

L2 regularization 이 포함된 손실함수에 SGD를 적용한 weight 업데이트 식은 다음과 같다.

![image-20200915141935325](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D5.png?raw=true)

이때, L2 regularization 이 포함된 손실함수  ftreg(θ)를 편미분하면 다음과 같다고한다.

![image-20200915142022604](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D6.png?raw=true)

이후 weight 업데이트 식에 대입하면  다음과 같다.

![image-20200915142058340](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D7.png?raw=true)

만약, λ′=λ/α 라면 L2 regularization은 정확히 weight decay와 같은 역할을 한다. 여기서 주목해야할 또 다른 포인트는 λ′=λ/α에서 regularization 상수 λ′이 learning rate α에 dependent하다는 것이다. 만약 사용자가 일반화 능력이 가장 좋은 regularization 상수 λ′을 찾았다고 하자. 그런데 이 때 learning rate α를 바꾸면 더 이상 λ′가 최적의 하이퍼파라미터가 아닐 수도 있다는 것을 의미한다.



**Adam**

Adam 은 gradient 의 1차 모먼트 m(t) 와 2차 모먼트 v(t) 를 사용하여 모멘텀 효과와 weight 마다 다른 learning rate 를 적용하는 adaptive learning rate 효과를 동시에 보는 최적화 알고리즘이다.

![image-20200915150734334](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D8.png?raw=true)

Adam은 위와 같이 weight 업데이트를 해준다. Adam에서는 L2 regularization 과 weight decay가 다르다는 것을 보이기 위해 weight 업데이트 식을 다음과 같이 간략하게 표현해 보면 다음과 같다한다.

![image-20200915150830210](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D9.png?raw=true)

SGD에서 구한   ∇fregt(θ) 을 대입한다면 아래와 같고

![image-20200915150916057](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D10.png?raw=true)

반면 weight decay 만 적용한 weight 업데이트 식은 다음과 같다.

![image-20200915151004025](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D11.png?raw=true)

이 둘은 M(t) = kL 가 아닌 이상은 같지 않을것이라고 하며  λ′ 앞에 M(t)가 붙기 때문에 SGD 경우보다  λ′ 가 더 작은 decay rate 로 weight decay 역할을 하게 된어 일반화 능력이 SGD보다 작게 된다고 한다.

## SGDW 와 AdamW

위에서 L2 regularization 을 사용하면 weight decay 효과를 볼수 있다는 것을 확인했다. 그리고 Adam 의 경우 그 효과를 덜 받게 되는 것도 확인해 보았다. 논문의 저자는 L2 regularzation에 의한 weight decay 효과 뿐만 아니라 weight 업데이트 식에 직접적으로 weight decay 텀을 추가하여 이 문제를 해결한다. L2 regularization 과 분리된 weight decay 라고 하여 decoupled weight decay 라고 말하는 것이다.

![image-20200915151337695](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2019-12-29-paper_review_AdamW/figure3.PNG)



![image-20200915151359687](https://raw.githubusercontent.com/HiddenBeginner/hiddenbeginner.github.io/master/static/img/_posts/2019-12-29-paper_review_AdamW/figure4.PNG)

SGDW 와 AdamW 알고리즘이다. η(7번,11번 라인) 가 있는데 이는 매 weight 업데이트마다 learning rate 를 일정 비율 감소시켜주는 learning rate schedule 상수를 의미한다. 초록색으로 표시된 부분이 없다면 L2 regularization을 포함한 손실함수에 SGD와 Adam을 적용한것과 똑같다. 하지만 초록색 부분을 직접적으로 weight 업데이트 식에 추가시켜줌으로써 weight decay 효과를 볼수 있게 만들었다.



## 논문 에서 실험

저자는 Shake-Shake regularization 에서 수행한 실험을 벤치마킹하여 자신의 가설들을 검증하였다.

**사용한 데이터**

* CIFAR-10
* ImageNet의 다운샘플버전(32*32 이미지 120만장)

**모델 구조** 

shake-shake regularization 모델(feature map을 augmentation 해주는 residual branch라는 구조를 갖고 있다고함)

- 26 2x64d ResNet
  - 깊이 26
  - 2개의 residual branch
  - 첫 번째 residual branch의 filter 수 64개,
  - 총 파라미터 11.6M
- 26 2x96d ResNet
  - 깊이 26
  - 2개의 residual branch
  - 첫 번째 residual branch의 filter 수 96개,
  - 총 파라미터 25.6M

batch size=128

regular data augmentation

**서로 다른 learning rate schedule 에서의 Adam과 AdamW 성능비교**

이 실험은 L2 regularization 에 Adam을 적용했을때와 weight decay까지 추가한 AdamW의 성능을 비교하는 실험이다. 이 논문에서 일반적인 learning rate schedule에서 작동하는 알고리즘을 제안했기 때문에 다음과 같이 서로 다른 세가지 learning rate schedule에 대해서도 실험을 진행하였다

* fixed learning rate
* step-drop learning rate
* cosine annealing

![image-20200915135724136](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure1.png?raw=true)

**논문내용**

Adam 은 L2보다 분리된 가중치 감소(하단 행 AdamW)에서 더 잘수행한다. 정규화(맨위행 ,Adam) 

CIFAR-10 에서 26개의 2x64d ResNet의 최종 테스트 오류를 보여준다. 고정학습률(왼쪽 열)

단계적 학습률(중간열,감소 에퐄 인덱스 30,60,80) 및 코사인 어닐링(오른쪽 열).

 AdamW 리드 더 분리 가능한 초 매개 변수 검색 공간으로, 특히 다음과 같은 학습률 일정이 스텝 드롭 및 코사인 어닐링이 적용된다. 코사인 어닐링으로 탁월한 결과를 얻을수 있다.

> 코사인 어닐링 : 코사인 형태로 훈련비율(learning rate)을 차츰 조정하면서 좀더 정확도를 상승시킬수 있는 방법

**결과해석**

> * Adaptive learning rate를 사용하는 Adam은 weight 마다 다른 lr을 적용하기 때문에 lr schedule의 중요성이 떨어지는 것처럼 보이지만 적절한 lr schedule을 사용하면 더 좋은 성능을 얻을수 있다.
>   * 특히, cosine annealing 이 가장 좋은 성능을 보인다.
> * AdamW가 Adam보다 모든 lr schedule에 대해 좋은 성능을 얻었다.
> * AdamW의 하이퍼파라미터 공간이 더 잘 구분된다.
> * 이후 실험은 모두 cosine annealing을 적용한다.



**SGD vs SGDW, Adam vs AdamW**

![image-20200915152551128](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure2.png?raw=true)

**결과해석**

> * Decoupled weight decay를 적용하지 않고 오직 L2 regularization 만 사용할 경우(1열), 성능 상위 10개의 하이퍼 파라미터(검은색원)가 대각선으로 분포하는 것을 볼수있다. 이는 L2 regularization만 사용했을때 얻을수 있는 weight decay 효과가 learning rate에 종속되어 있기 때문이다. ((λ′=λ/α)
>   * 따라서, 최적의 하이퍼파라미터를 찾기 위해서는  α와 λ를 동시에 바꿔줘야 한다.
>   * 1행 1열의 가장 왼쪽 위의 검은 원 (α=1/2,λ=1/8∗0.0001α=1/2,λ=1/8∗0.0001)에서 HP 튜닝을 할 때, α나 λ 중 하나만 바꿔가면 성능이 더 안 좋아질 것이다. 더 좋은 성능을 얻기 위해서는 α와 λ를 동시에 바꿔주어야 한다
>   * SGD가 하이퍼파라미터에 민감하다는 평판이 있는데 이러한 이유때문이다.
> * 반대로 decoupled weight decay를 사용할 경우 learning rate와 weight decay가 서로 독립적이다. 이 말은 어느 한 하이퍼파라미터를 고정하고 다른 하나만을 바꿔가도 더 좋은 성능을 얻을수 있다는 뜻이다.
> * 둘다 L2 Regularization을 사용했는데도 Adam의 결과는 확실하게 SGD보다 안좋다.
>   * Adam은 λ=0λ=0일 때나 λ≠0λ≠0 일 때나 검은 원이 있는 성능이 비슷하다. 즉, Adam은 L2 regularization의 효과를 덜 받는다.
> * AdamW는 SGD와 SGDW와 필적할 만한 성능을 보였다.



**AdamW 와 Adam의 일반화 능력비교**

![image-20200915153220424](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure3.png?raw=true)

**실험환경**

CIFAR-10, 26 2x96d ResNet, Epochs 1800, learning rate = 0.001, normalized weight deacy 사용

**결과해석**

> * 학습 초기에는 Adam과 AdamW가 비슷한 loss를 보이지만 학습이 진행될수록 AdamW의 훈련손실과 test 에러가 더 낮아진다.
> * 2행 2열의 그래프에서 같은 훈련 손실에 대해서 AdamW의 테스트 에러가 더 낮은 것을 볼수 있다.
>   * 즉, AdamW의 좋은 성능이 단순히 학습동안 더 좋은 수렴 지점을 찾았기 때문이 아니라, 더 좋은 일반화 능력이 있기 때문이다.



**AdamWR**

![image-20200915154332166](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure4.png?raw=true)

> epochs 에 따른 ImageNet32x32 top 5 

Adam, AdamW, SGDW, AdamWR,SGDWR 순서로 성능이 좋은것이 보인다.

warm restart를 사용하는 AdamWR, SGDWR의 경우 테스트 에러가 학습 중간중간 뛰는 모습을 보이는데 이것은 학습 중간에 learning rate를 점프 시키기 때문이라고 한다. 이 작용으로 AdamWR 과 SGDWR은 일반화를 더 잘하는 local minimum을 찾을수 있다고 한다.



**Learning rate schedule 과 Learning rate annealing**

`Learning rate schedule` 란 단어 그대로 훈련 동안에 고정된 learning rate를 사용하는 것이 아니고 미리 정한 스케줄대로 learning rate를 바꿔가며 사용하는 것이다. 그리고 `learning rate annealing` 은 learning rate schedule과 혼용되어 사용되지만 특히, learning rate 가 iteration에 따라 monotonically decreasing 하는 경우를 의미하는것 같다. anneal은 `담금질하다` 라는 뜻을 가지고 있다. Learning rate annealing을 직영하자면 `학습률 담금질하기` 라고 생각된다. Learning rate annaeling을 사용하면 초기 learning rate를 상대적으로 크게 설정하여 local minimum에 보다 더 빠르게 다가갈수 있게 만들어주고 이후 learning rate를 줄여가며 local minimum에 보다 더 정확하게 수렴할수 있게 만들어준다.

![image-20200915154937219](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph1.png?raw=true)

> step-drop learning rate decay(왼쪽), linearly decreasing learning rate decay(오른쪽)

Learning rate annealing 에는 다양한 방법이 있을수 있다. 가장 쉬운 예로, 위의 왼쪽 그림처럼 learning rate를 학습이 진행됨에 따라 step function 처럼 감소시킬수 있다. 이를 `step-drop learning rate decay` 라고 한다. 또는 오른쪽 그림처럼 학습이 진행됨에 따라 learning rate를 선형적으로 감소시킬 경우 `linearly decreasing learning rate decay` 라고 한다. 그리고  `cosine annealing` 은 아래 그림처럼 half-consine 그래프를 따라 learning rate를 감소시킨다.

![image-20200915155209959](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph2.png?raw=true)

직관적으로 처음에는 높은 learning rate로 좋은 수렴 지점을 빡세게 찾고, 마지막에는 낮은 learning rate로 수렴 지점에 정밀하게 안착할수 있게 만들어주는 역할을 할것 같다. 여기까지가 learning rate annealing에 대한 설명이다. 복습하자면 learning rate annealing은 iteration에 따라 learning rate 를 감소시켜주는 방법이라고 한다. 하지만 훈련도중 learning rate를 증가시켜주는 learning rate schedule을 사용할수도 있다. 



**Warm restart**

좋은 수렴 지점에 대한 직관적 이해

일반화 능력이 좋은 모델은 훈련 데이터의 분포와 조금 다른 분포의 테스트 데이터가 들어와도 역할을 잘 수행할수 있는 모델이다. 아래 그림처럼 weight에 따른 loss함수의 그래프가 있다고 하고 훈련을 통해 최적의 weight w(1)을 찾았다고 생각해보면

![image-20200915155526178](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph3.png?raw=true)

위 그림의 갈색선은 "훈련 데이터로 부터 만들어진 weight에 대한 loss함수"의 그래프이다. "훈련 데이터"로 부터 만들어진 loss 함수이기 때문에 훈련 데이터의 분포와 다른 분포의 테스트 데이터에 대해서는 다른 loss 함수가 만들어 질것이다. 예를 들어 아래 그림처럼 테스트 데이터의 분포가 훈련 데이터의 분포와 달라서 다음 그림과 같이 훈련 데이터(살구색)과 테스트데이터(갈색)의 loss 함수 그래프가 다르게 나타났다고 보면

![image-20200915155813293](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph4.png?raw=true)

테스트 데이터에 대한 loss 함수는 훈련 데이터에 대한 loss 함수에 비해 아주 조금 달라졌지만, w(1) 에서의 테스트 데이터의 loss는 훈련 데이터의 loss 와 굉장히 차이나게 된다. 즉 w(1) 처럼 가파른 local minimum에서는 테스트 데이터의 분포가 조금만 달라져도 error 가 민감하게 변한다는 의미이다. 반대로, 모델이 훈련을 통해 최적의 weight를 w(2)로 찾았다고 생각해보자

![image-20200915155941505](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph5.png?raw=true)

하지만 이 경우 테스트 데이터에 대한 loss 함수가 훈련 데이터에 대한 loss 함수에 비해 달라졌다해도 w(2)에서의 테스트 데이터의 loss는 훈련 데이터의 loss와 크게 달라지지 않는다.

![image-20200915160031240](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph6.png?raw=true)

이런 평평한 지점의 weight들은 훈련 데이터의 분포와 다른 테스트 데이터가 들어와도 상대적으로 안정적인 loss값을 얻을수 있다. 즉, 보다 일반화 되었다고 말할수 있다

**Warm restart**

warm restart는 위와 같은 문제를 해결하기 위한 한가지 방법이다. 학습 중간중간에 learning rate를 증가시켜 큰 폭의 weight update 를 만들어 가파른 local minimum에서 빠져나올 기회를 제공한다.

예를 들어 위의 learning rate schedule은 initial learning rate를 0.01로 설정하고 cosine annealing을 사용하며 iteration 380과 iteration 760에서 learning rate를 증가시키는 warm restart이다. 이 learning schedule을 사용하여 다시 한번 위의 예시를 살펴보자면 먼저 inital weight 지점에서 시작하여 gradient descent를 통해 local minimum w(1)에 수렴하였다. 업데이트 폭이 점점 줄어드는 것은 cosine annealing을 사용한 것으로 볼수 있다.

![image-20200915160338658](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph7.png?raw=true)

이때 cosine annealing만 사용하였다면 learning rate가 0에 가까워져 w(1)에서 weight 업데이트가 중지되었을 것이다. 하지만 다시 learning rate를 증가시키면 아래 그림처럼 가파른 local minimum을 탈출할수 있게 된다. learning rate가 크기 때문에 업데이트 폭도 큰것을 볼수 있다.

![image-20200915160447985](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph8.png?raw=true)

다시 cosine annealing에 따라 learning rate를 줄여가며 gradient descent를 하면 아래 그림처럼 w(2)로 가게 된다.

![image-20200915160531676](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/graph9.png?raw=true)

learning rate 증가시키는 iteration 지점을 2곳으로 설정했기 때문에 한번더 learning rate가 증가되게 된다. 하지만 이번에는 업데이트 폭이 커도 같은 local minimum 안의 weight로 업데이트가 되게 된다.

![image-20200915160635081](C:\Users\s_m04\OneDrive\문서\paper-revice\PR-003\image\graph10.png)



**Normalized weight decay**

논문 저자는 최적은 weight decay 상수 λ가 총 weight update 횟수에 종속적이라는 것을 실험을 통해 확인하였다. 아래 그림은 CIFAR-10 을 25, 100,400 번 학습했을때 test error 이다. 성능 상위 10개의 hyperparameter setting 이 검은색 원으로 표시되어있다. epoch이 작을수록 최적의  λ값은 크고, epoch이 클 수록 최적의 λ 값이 작다는 것을 확인할 수 있다.

![image-20200915164026691](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure5.png?raw=true)

파라미터 업데이트 횟수에 따라 최적의  λ 값이 달라진다면 하이퍼파라미터튜닝하기가 더 어려울 것이다. Normalizaed weight decay는 파라미터 업데이트 횟수에 따라 사용할  λ 값을 정하는 방법이다. 

사용자는 하이퍼파라미터로 λλ 대신 λnormλnorm이라는 것을 설정해준다. 다음으로 λ=λnorm√bBTλ=λnormbBT 으로 계산한 λλ를 weight decay 상수값으로 사용하게 된다. 여기서 bb는 batch size, BB는 훈련 데이터 개수, TT는 총 에폭 횟수다. 예를 들어,

배치 사이즈가 클수록, 데이터가 적을 수록, Epoch 수가 적을 수록 총 파라미터 업데이트 횟수는 적다.

![image-20200915164159944](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D12.png?raw=true)

배치 사이즈가 작을 수록, 데이터가 많을 수록, Epoch 수가 많을 수록 총 파라미터 업데이트 횟수는 많다.

![image-20200915164228996](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/%EC%88%98%EC%8B%9D13.png?raw=true)

사용자가 λnormλnorm만 설정해주면 총 파라미터 업데이트 횟수에 따라 사용할 λ 값을 자동으로 선택해지기 때문에 Hyperparameter 튜닝하기가 한결 쉬워진다.

![image-20200915164309856](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure6.png?raw=true)



**AdamWR vs SGDWR vs AdamW vs SGDW vs Adam**

![image-20200915164352497](https://github.com/roche-MH/paper-review/blob/master/PR-003/image/figure7.png?raw=true)

**실험환경**

- 1열: Epoch에 따른 Top-1 test error and training loss on CIFAR-10
- 2열: Epoch에 따른 Top-5 test error and training loss on ImageNet32x32
- 어떤 모델을 사용했는지는 언급되지 않았지만 CIFAR-10에서 1,800 Epoch인걸보아 실험 3에서 사용한 26 2x96d ResNet일 것 같다

**결과해석**

* 마지막 epoch 에서 학습을 마쳤을때 AdamWR 과 SGDWR의 training loss가 다른 알고리즘들에 비해 높지만 test error은 더 낮다. 즉 AdamWR과 SGDWR이 더 generalization을 잘한다는것 같다.

**마지막 정리**

* Adaptive gradient methods 들은 L2 regularzation 에 의한 weight decay 효과를 온전히 볼수 없다.
* L2 regularziation에 의한 weight decay 효과와 별개로 weight decay를 weight업데이트 식에 넣어주었다(decoupled weight decay)
* Learning rate schedule이 Adam의 성능 상승에 도움을 줄수 있다는 것을 확인하였다.
* Warm restart 까지 적용한다면 성능은 더 좋아질것이다.

