# Generative Adversarial Nets 리뷰

GAN (Generative Adversarial Network) 이 세글자의 뜻을 풀어 보는 것만으로도 GAN에 대한 전반적 으로 이해할수 있다.

Generative 는 생성 모델이라는 것을 뜻한다. 생성 모델이란 '그럴듯한 가짜'를 만들어내는 모델을 말한다. 

Adversarial 은 GAN 이 두개의 모델을 적대적으로 경쟁시키며 발전 시킨다는 것을 뜻한다.

Network 는 모델이 인공신경망(Artificial Neural Network) 또는 딥러닝으로 만들어진 것을 뜻한다.



## Taxonomy of ML

![image-20200720225549225](https://www.researchgate.net/profile/Christian_Esteve_Rothenberg/publication/335793747/figure/fig1/AS:802663088287744@1568381201676/A-taxonomy-of-mainstream-ML-approaches.png)

머신러닝을 크게 분류 하자면 Supervised , Unsupervised, Reinforcement 로 분류 할수 있다. 이중 GAN 은 Unsupervised Learning 에 속하고 비지도 학습에서는 [Boltzmann machine(RBM)](http://sanghyukchun.github.io/75/) , [Auto-encoder](https://excelsior-cjh.tistory.com/187) , GAN 이 있다.

Auto-encoder 의 수식에서 크게 보았을때 KL-divergence + Likelihood  가 되는데 Liklihood 는 Linear regression(MSE) 과 비슷해서 이미지 생성시 이미지가 많이 blur 한 부분이 많기 때문에 GAN 에서는 Adversarial 를 사용하여 뚜렷한 이미지를 만든다고 한다.

![image-20200720231603153](https://i.imgur.com/JnoyZIN.png)

> Likelihood : 확률 변수 x 에 대한 확률 모형은 확률 밀도 함수 f(x) 에 의해 정의된다.
>
> KL-divergence : 두 확률 분포의 차이를 계산하는 함수로, 어떤 이상적인 분포에 대해, 그 분포를 근사하는 다른 분포를 사용해 샘플링을 한다면 발생할수 있는 정보 엔트로피 차이



![image-20200720234627900](https://files.slack.com/files-pri/T25783BPY-F9SHTP6F9/picture2.png?pub_secret=6821873e68)

> Generator  = 임의의 랜덤 데이터를 받아들여 진짜같은 가짜 데이터를 생성
>
> Discriminator = 실제 데이터, 가짜 데이터를 판별 (real 1, fake 0 출력)
>
> Fake data = Qmodel(X|Z) (랜덤값Z를 줬을때 X이미지를 내보내는 모델)
>
> Real data = Pdata(X)



GAN 모델의 loss function

GAN 네트워크가 D,G 모두가 최선의 목적을 달성할수 있도록 한다고 했다.

GAN 네트워크에서의 학습이 다른 신경망과 달리 한가지의 최적화(min or max) 만을 위한것이 아님을 드러내며 다음과 같은 loss function 을 사용한다.

![image-20200720234422645](https://github.com/roche-MH/paper-review/blob/master/PR-001/image/Diagram.PNG?raw=true)



Discriminator 을 보면 D는 V(D,G)를 최대화 하도록 학습한다.

* D(x) 가 1일때 진짜 데이터를 진짜라고 판별해야하고
* D(G(z)) 가 0일때 : 가짜 데이터를 가짜라고 판별해야한다.



Generator 은 V(D,G) 를 최소화 하도록 학습한다.

* D(x)는 무관하고 D가 진짜 데이터를 진짜라고 판별하는것에 개의치 않고
* D(G(z)) 가 1일때 D가 가짜 데이터를 진짜라고 판별해야 한다.



![image-20200720234842803](https://github.com/roche-MH/paper-review/blob/master/PR-001/image/discriminator.PNG?raw=true)

> 파란색 : Discriminator
>
> 검은색 : discriminator data
>
> 초록색 : real data

검은 선이 P 이고 초록색이 Q 라고 했을때 generator 는 P를 Q 와 동일하게 하고자 학습을 이어나 간다. 그리고 real 과 fake 이미지가 동일시 되었을때 Discriminator 는 1/2 확률로 분류할수 밖에 없다고 한다.



# Theorem

p(g) = pdata 가 같을때  D*G(x) = 1/2 이 되고  1/2 를 minmax에 넣게 되면 다음과 같이 수식이 진행된다.

![image-20200721001843266](https://github.com/roche-MH/paper-review/blob/master/PR-001/image/log4.PNG?raw=true)

![image-20200721001743195](https://github.com/roche-MH/paper-review/blob/master/PR-001/image/CG.PNG?raw=true)

> C(G) 는 D를 MAX 로 고정해 두었을때 수식

![image-20200721001948039](https://github.com/roche-MH/paper-review/blob/master/PR-001/image/JSD.PNG?raw=true)

> KL-divergence를 2개 더하게 되면 JSD 와 동일하게 된다

C* = -log(4) 가 C(G) 의 global minimum 이고 그 유일한 해가 pg= pdata가 된다.

