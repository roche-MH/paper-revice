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

Adversarial Diagram of  Standard GAN

![image-20200720234422645](../../../../AppData/Roaming/Typora/typora-user-images/image-20200720234422645.png)



![image-20200720234627900](https://files.slack.com/files-pri/T25783BPY-F9SHTP6F9/picture2.png?pub_secret=6821873e68)

> Generator  = 생성자
>
> Fake data = Qmodel(X|Z) (Z값을 줬을때 X이미지를 내보내는 모델)
>
> Real data = Pdata(X)
>
> Discriminator = 판별자

![image-20200720234842803](../../../../AppData/Roaming/Typora/typora-user-images/image-20200720234842803.png)

> 파란색 : Discriminator
>
> 검은색 : discriminator data
>
> 초록색 : real data

검은 선이 P 이고 초록색이 Q 라고 했을때 generator 는 P를 Q 와 동일하게 하고자 학습을 이어나 간다. 그리고 real 과 fake 이미지가 동일시 되었을때 Discriminator 는 1/2 확률로 분류할수 밖에 없다고 한다.