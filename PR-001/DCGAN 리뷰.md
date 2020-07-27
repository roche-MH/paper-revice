# DCGAN 리뷰



## 기존 GAN의 한계

1. GAN은 결과가 불안정하다

기존 GAN 만 가지고는 좋은 성능이 잘 안나왔다.

2. Black-box method

Neural Network 자체의 한계라고 볼수 있는데, 결정 변수나 주요 변수를 알수 있는 다수의 머신러닝 기법들과 달리 Neural Network 는 처음부터 끝까지 어떤 형태로 그러한 결과가 나오게 되었는지 그 과정을 알수가 없다.

3. Generative Model 평가

GAN은 결과물 자체가 새롭게 만들어진 Sample 이다. 이를 기존 sample 과 비교하여 얼마나 비슷한지 확인할수 있는 정량적 척도가 없고, 사람이 판단하더라도 이는 주관적 기준이기 때문에 얼마나 정확한지, 혹은 뛰어난지 판단하기 힘들다.



## DCGAN 의 목표

> 1. Generator 가 단순 기억으로 generate하지 않는다는것을 보여줘야한다.
> 2. z의 미세한 변동에 따른 generate결과가 연속적으로 부드럽게 이루어져야 한다(walking in the latent space 라고 함)


## Architecture Guidenlines

GAN과 DCGAN 의 전체적인 구조는 거의 유사하지만, 각각의 Discriminator 와 Generator의 세부적인 구조가 달라진다. 논문에서는 이 구조를 개발한 방법을 

이렇게 말했다 

> after extensive model exploration we identified a family of archi-tectures that resulted in stable training across a range of datasets and allowed for training higher
> resolution and deeper generative models.

다양한 테스트를 거쳐 높은 결과 값을 가지는 parameter 를 추출해 냈다는것 같다.



**기존 GAN Architecture**

기존의 GAN은 간단한 fully-connected로 연결되어 있다.

![image-20200727180421830](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/gan-architecture.png)

**CNN Architecture**

CNN은 이러한 fully-connected 구조 대신에 convolution, pooling,padding을 활용하여 레이어를 구성한다.

![image-20200727180539026](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/cnn-architecture.png)



**DCGAN Architecture**

DCGAN은 결국, 기존 GAN에 존재했던 fully-connected 구조의 대부분을 CNN 구조로 대체한 것이다.

![image-20200727180659148](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/architecture-guidelines.png)

* Discriminator 에서는 모든 pooling layers를 strided convolutions로 바꾸고, Generator 에서는 pooling layers를 fractional-strided convolution으로 바꾼다.

> Strided convolutions
>
> ![image-20200727183337816](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/padding_strides.gif)
>
> 파란색이 input, 초록색이 output
>
> * Kernel Size : Kernel size 는 convolution의 시야(view)를 결정한다. 보통 2D에서는 3x3 pixel 로 사용한다.
> * Stride : Stride는 이미지를 횡단할때 커널의 스텝 사이즈를 결정한다. 기본 값은 1이지만 보통 Max Pooling 과 비슷하게 이미지를 다운샘플링하기 위해 Stride를 2로 사용할수 있다.
> * Padding : Padding 은 샘플 테두리를 어떻게 조절할지를 결정한다. 패딩된 Coonvolution은 input 과 동일한 output 차원을 유지하는 반면, 패딩되지 않은 Convolution은 커널이 1보다 큰 경우 테두리의 일부를 잘라버릴수 있다.
> * Input & Output Channels : Convolution layer 는 input 채널의 특정수(l) 을 입력받아 output 채널의 특정 수 (O)로 계산한다. 이런 계층에서 필요한 파라미터의 수는 I * O* K 로 계산할수 있다. (K는 커널수)



> Fractionally-strided convolutions
>
> ![image-20200727183458376](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/padding_strides_transposed.gif)
>
> Convolution 작업을 하면서 5x5 이미지의 output을 생성하는 것, 이 작업을 하기 위해 input에 임의의 padding 을 넣어야 한다.
>
> 단순히 이전 공간 해상도를 재구성하고 convolution을 수행한다. 수학적 역 관계는 아니지만 인코더-디코더 아키텍쳐의 경우 유용하다. 이 방법은 2개의 별도 프로세스를 진해하는 것 대신 convolution된 이미지의 upscaling 을 결합할수 있다.
>
> Transposed Convolution는 일반적인 convolution을 반대로 수행 하고 싶은 경우 사용하며, 커널 사이에 0을 추가한다.1
>
> ![image-20200727184326256](https://www.dropbox.com/s/ksuhdbpji514bqm/Screenshot%202018-06-04%2021.46.09.png?dl=1)

* Generator 와 Discriminator 에 batch-normalization을 사용한다. 논문에서는 이를 통해 deep generators의 초기 실패를 막는다고 한다. 그러나 모든 layer에 다 적용하면 sample oscillation과 model instability의 문제가 발생하여 Generator output layer 와 Discriminator input layer에는 적용하지 않는다고 한다.

> Batch Normalization은 기본적으로 Gradient Vanishing / Gradient Exploding 이 일어나지 않도록 하는 아이디어중 하나이다.
>
> * Gradient Vanishing, Gradient Exploding : 기울기값이 사라지는 문제
>
> Batch Normalization 
>
> * Batch Normalization 에서는 각 layer 에 들어가는 input을 normalize 시킴으로써 layer 의 학습을 가속하는데, 이때 whitening 등의 방법을 쓰는 대신 각 mini-batch 의 mean 과 variance 를 구하여 normalize 한다.
> * 각 feature 들이 uncorrelated 되어 있다는 가정의 효과를 없애주기 위해, scaling factor 와 shifting factor 를 layer 에 추가로 도입하여 back-prop으로 학습시킨다.

* Fully-connected hidden layers를 삭제한다.

> Fully-connected : 이전 레이어의 모든 노드가 다음 레이어의 모든 노드에 연결된 레이러를 Fully Connected Layer 라고 하면 Dense Layer 라고도 한다.

* Generator에서 모든 활성화 함수를 Relu를 쓰되, 마지막 결과에서만 Tanh를 사용한다.

> Generator 에서 ReLU와 마지막 output Tanh 를 쓰는 이유는
>
> 경계가 있는 activation 이 모델이 학습을 더 빨리 수렴하도록 학습하고 학습 분포의 색공간을 빠르게 커버하도록 하기 때문이라고 한다.

* Discriminator에서는 모든 활성화 함수를 LeakyRelu를 쓴다.



## Generator Model

가이드라인의 Generator를 시각화 하면 다음과 같다.

![image-20200727191600098](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/generator-model.png)

100 dimensional uniform distribution(Z) 가 들어오면 이들이 4개의 fractionally-strided convolution layer 을 거치며 크기를 키워서 더 높은 차원의 64x64 pixel 이미지가 된다.



# black Box

![image-20200727192905968](https://angrypark.github.io/images/2017-08-03-DCGAN-paper-reading/visualization-3.png)

* Network 에 convolution을 사용하게 되면서 3가지의 문제점을 해소하였다.
  * decocnvolution 을 통한 high resolution 한 이미지를 생성할수 있다
  * filter visulization 통해 시각화해서 어떻게 만들고, 구별하는지 인지하며, 기존의 Black Box 라는 문제점을 해소할수 있었다.
  * Convolution을 통해 이미지의 특징을 학습하여 다양하고 실제같은 이미지를 학습할수 있고 안정적인 학습이 가능하다.

