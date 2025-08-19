# US Stock Market News Sentiment Analysis
이 프로젝트의 목표는 미국 증시 뉴스 내용을 기반으로, 해당 뉴스에서 언급된 기업 또는 특정 시장 상황을 약세(Bearish), 중립(Neutral), 강세(Bullish)로 분류하는 미국 증시 도메인 특화 모델을 구축하는 것입니다.

사용 가능한 GPU 메모리가 한정되어 있기 때문에, 공개된 대규모 사전학습 모델 대신 작은 크기의 사전학습 모델을 구축하고자 하였으며, 최종적으로 T5_samll 크기의 사전학습 모델을 만들었습니다.

이 사전학습 모델을 이용하여 두 가지 방식의 파인튜닝을 진행했습니다. 
- 첫 번째는 일반적인 classification task에서 많이 사용되는 방법으로, 모델에 classification head layer를 추가하여 파인튜닝하는 방식입니다.
- 두 번째는 T5처럼 text-to-text로 파인튜닝하는 방식입니다. 이를 위해 'sentiment: '라는 prefix를 사용했습니다.

## 1. datasets
파인튜닝에 사용할 텍스트 데이터셋은 야후 파이낸셜 뉴스이며, 각 뉴스에 Bearish/Neutral/Bullish 레이블이 붙어 있는 지도학습 데이터셋입니다. (https://huggingface.co/datasets/ugursa/Yahoo-Finance-News-Sentences/viewer/default/train?views%5B%5D=train&row=10)

사전학습에서도 파인튜닝 데이터셋과 동일한 도메인 데이셋(블룸버그 파이낸셜 뉴스)을 사용하여, 주식/금융 분야에서 자주 등장하는 복합어 및 특수 용어를 모델에 노출시킴으로써, 해당 도메인 언어 패턴에 익숙하게 만들어 이후 파인튜닝 단계에서 성능 향상을 기대할 수 있습니다. (https://huggingface.co/datasets/genloop/bloomberg_financial_news_120k)

두 데이터셋에 모두 휴리스틱한 텍스트 전처리를 진행하였습니다. 

비지도 사전학습을 위한 데이터셋 처리에 대한 내용은 <code>prepare_unsupervised_dataset_for_pretraining.ipynb</code>에서, 파인튜닝을 위한 데이터셋 처리에 대한 내용은 <code>prepare_supervised_dataset_for_finetuning.ipynb</code>에서 확인할 수 있습니다. 

## 2. pre-training
사전학습 모델은 Google에서 발표한 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" 논문(https://arxiv.org/abs/1910.10683)에서 제안된 T5(Text-to-Text Transfer Transformer) 모델의 방법론을 기반으로 합니다. 

- (1) 시퀀스 패킹(sequence packing)
- 논문에서는 패딩 토큰으로 인한 연산 낭비를 최소화하기 위해 시퀀스 패킹을 적용하였습니다.
- 이 프로젝트에서도 이러한 낭비를 줄이고자, 여러 입력 토큰 ID를 이어 붙여 시퀀스 최대 길이(512)를 채우는 방식을 사용했습니다. 
  
  



### 2.3 slim attention

### 2.4 t5

<div align="center">
  <img width="350" height="150" alt="image" src="https://github.com/user-attachments/assets/40b5a0d0-a921-404c-b5d5-4befa2788d5f" />
</div>


<div align="center">
  <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/f687b12f-bc84-4f97-b48f-7d3bc90a1b3a" />
</div>


## 3. fine-tuning

<div align="center">
  <img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/0be484dc-dd7b-43ce-97da-3cedd86672bf" />
</div>

### 3.1 qlora

<div align="center">
  <img width="285" height="250" alt="image" src="https://github.com/user-attachments/assets/6f80cbc3-6160-405f-ac6a-f92cba474f09" />
</div> 

### 3.2 classification head fine-tuning

<div align="center">
  <img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/148239b3-74d5-42ff-a4d4-9261c199a3b7" />
</div>


### 3.3 text-to-text fine-tuning

<div align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/0e8d3bf9-7ab2-4bb2-b66b-49e766000a8d" />
</div>
