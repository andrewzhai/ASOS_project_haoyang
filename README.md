# About the project

In this project I am going to implement and evaluate the different strategies for data augmeantion and confident learning in our NLU training data.

The code could achieve the following objectives
+ Data Augmentation in Transformers Library with T5 and GPT3 models.
+ Remove misclassified data with confident learning
+ Training and Testing NLU classifers based on the modeified datasets(Data augmenation with different sizes/Data augmenation and Conifident learning)
+ Evaluate the different Data augmenation and confident learning stratgies based on the performance of relative NLU classifier   


# Before you start

## Execute in Google Colab

To get a better demonstration to my work, I remcommend you open those scripts in Google Colab and excute all cells in order.

Colab can load public github notebooks directly, with no required authorization step.


To generate such links in one click, you can use the [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) Chrome extension.

## installation
Before you start, here are some of required packages you might need to install later. But dont worry, I provide the installation code in my Google Colab notebook as well. It would be available in google colab files via pip
```bash
pip install [Package you need]
```
### Set up Git
```bash
git clone https://github.com/andrewzhai/ASOS_project_haoyang.git

```
### Main packages required
Note: Rasa framework sometimes could be quite delicate to version of packages you installed, in order to recreate the exact same results, I suggest excute all my codes in order and in google colab, I provided a quick link to open .ipynb files in google colab

+ PyYAML==5.1.2
+ transformers==2.8.0
+ torch
+ numpy==1.19.0
+ sklearn
+ prompt_toolkit==2.0.1
+ rasa
+ json
+ cleanlab
+ sentence-transformers


# Usage and Instruction

## Guide
In this project, I provided three different scripts demonstraing in Google Colab Notebooks to achieve different objectives.
For different applictations you could excute and modified different colab notebooks yourself. For example, if you want to generate data you can use [T5 and Gpt3 generator.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/T5%20and%20Gpt3%20genator.ipynb), if you want to prune your augmented data you can use [confient_learning.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/confident_learning.ipynb), and for the rasa training and testing you could use [Data_augmentaion_and_confient_learning_RASA_evaluation.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/Data_augmentaion_and_confient_learning_RASA_evaluation.ipynb)

## Data Augmentation with T5 and GPT3 models

[T5 and Gpt3 generator.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/T5%20and%20Gpt3%20genator.ipynb)
In this section, data augmentation is achieved with the use of Transformers library and different pre-trained models(T5,GPT3) For the further comparison, we are going to compare the performance of the NLU model with the amount of augmentation data.As shown in the following figure,the augmentation data is only generated based on the training set (we generate from 0 data, to 10 data).

![alt text](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/output_result_by_haoyang/figure/1.jpg)

[T5 and Gpt3 generator.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/T5%20and%20Gpt3%20genator.ipynb) is able to achieve several different objectives:
+ data preprocessing and apply 5-fold cross validation


+ T5 data augmentation with different amount of newly generated sentences

This function is developed based on the transformers library with T5 pre trained model.
```bash
def generate_sentences(sentence,num,k,p): 


  text =  "paraphrase: " + sentence + " </s>"
  print(text)


  max_len = 256

  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

             
  
  beam_outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      do_sample=True,
      max_length=256,
      top_k=k,
      top_p=p,
      early_stopping=True,
      num_return_sequences=30
  )
  final_outputs=[]
  for beam_output in beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      if sent.lower() != sentence.lower() and sent not in final_outputs:
          final_outputs.append(sent)

  return final_outputs[0:num]
```

+ This part is to read 5 fold of yaml files and complete data processing 

at first I read the 5 original training files(1cv_original_train.yml,...5cv_original_train.yml) and did some basic data cleaning such as removing the extra dash symbols and empty lines. Then save the clean text in a (33,10) nested array named as’f_train’. We have 33 intents and each intent contains around 10 examples. We generate data with the first NLU training file, then we follow the same steps(as shown in the following workflow) for the rest of the training files. First of all, iterate array f_train, input one text for T5 data generation each time, for every input sentence, the T5 generator outputs 10 different paraphrased sentences(synthetic data). Then we sliced the output array in 10 different sizes: array[:1], array[:2]...array[:10], In this way we get 10 different arrays. Regenerate all the different sentences under each intent, and store them in an (33,10,10) array named ‘store’. Each element in this array contains one original sentence, and X more synthetic sentences. X variety from 1 to 10. The first element in this tuple, 33 represents the amount of intents,  the second position in this tuple represents that there are 10 different sentences in each intent, the last position in this tuple represents that there are 10 different-sized arrays containing different amounts of synthetic data.

The benefit of generating text following this workflow is to reduce the noise generated in the T5 data augmentation. As indicated in the workflow, each sentence was input into the T5 generator for only once. In other words, there are only two synthetic data different in file 6 and file 8 for each sentence. 


![alt text](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/output_result_by_haoyang/figure/Blank%20diagram.png)


```bash
for qq in range(1,6):
  #input 5 different files
  import yaml
  with open('/content/ASOS_project_haoyang/output_result_by_haoyang/data/original_data_5fold/{}cv_original_train.yml'.format(qq),'r') as f:
    data=yaml.safe_load(f)


  f_train=[]
  for i in data['nlu']:
    text=[]
    for j in i['examples'].split('\n'):     # get the clean data from yaml
      input=j.replace('- ','')
      text.append(input)
    f_train.append(text[:-1])
```



```bash
  c=data
  store=np.ones(32).tolist()
  import numpy as np
    

  count=0
  for i in f_train:
  
    arr=[]
    for sentence in i:
        l=[]
        # T5 data generator
        sentences=generate_sentences(sentence,12,150,0.87)
        print('----------------')
        print(sentence)
        l.append(sentence)
        l.extend(sentences) 
        new=[]
        print(l)
        ## store sentence
        for q in range(1,11):
            
          new.append(l[0:q+1])
        arr.append(new)
        
        
    store[count]=arr
    count=count+1


```
We read the stored array to write in 10 different NLU training yaml files.Then We repeat the same process for the rest 4 original files, in this way we have the 50 different NLU training yaml files. 


```bash
  d=[]
  for i in store:
    print(np.array(i).shape )
    d.append(np.array(i).shape[0])   



  for num in range(0,10):
    c=data
      
    count=0
    for (i,j,n) in zip(f_train,c['nlu'],d):
      new=[]
      print('*************')
      print(j['intent'])
      for q in range(n):
        new.extend(store[count][q][num])
          
            
          
      
        
        
      count=count+1
      print('-----------')
      print(len(new))
      j['examples']=new
        
    print(c['nlu'])
      

    with open('augmented_data_t5/{}cv_{}augumented_fixed.yml'.format(qq,num+1), 'w') as outfile:
          yaml.safe_dump(c, outfile, default_flow_style=False, sort_keys=False)
```

+ access Gpt3 augmentation and add different amount of newly generated sentences in each fold of original training data



## Confident learning
I introduced two different methods for confident learning in this scripts: [confient_learning.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/confident_learning.ipynb)
###  Method 1  Confident learning with Multiple-layer-perceptron
  +  MLP is considered as a suitable choice to train the CL-MLP classifier.Then augmented datasets are input as test sets in our trained CL-MLP classifier, so as to compare the labels predicted by MLP model with the current labels in the augmented data. If the predicted label by the CL-MLP classifier is different from the current label for each sentence, then we regard the indices of those input sentences as noisy labels.

train a MLP classifer clf with the only original file

```bash
from sklearn.neural_network import MLPClassifier
x_train,y_train,_=BertEmbed('/content/ASOS_project_haoyang/input_data/T5_nlu.yml')

clf = MLPClassifier(random_state=1, max_iter=300, solver='lbfgs')
clf.fit(x_train, y_train)
```

use the trained model clf to predict the probability for different sentences in other augmented data
```bash
x_test,y_test,_=BertEmbed('/content/ASOS_project_haoyang/output_result_by_haoyang/data/cv_T5_augmented_data/1cv_1augumented_fixed.yml')
result=clf.predict_proba(x_test)
```

set the threshold based on the returned probability to determine if this data is misclassifed
```bash
noise_ind=[]
count=0
for (i,j) in zip(result,y_test):
  if i[j]<0.8:
    noise_ind.append(count)
  count=count+1
print(noise_ind)
```
###  Method 2 Confident learning with Clean Lab package

Workflow: This workflow demonstrates the application of Clean Lab powered by Confident learning. First, generate the training dataset X in the form of matrix with Sentence Embedding and store the encoded noise intent labels as a target .Secondly, compute confident joints with clean lab built-in function. Finally, return the indices of noisy-data and regenerate the dataset by removing the noisy data.

I define this function with the input of augmented data and return the confident joint 
```bash
def computepsx(path):
  import yaml
  with open(path,'r') as f:
    data=yaml.safe_load(f)

  f_train=[]
  for i in data['nlu']:
    text=[]
    for j in i['examples'].split('\n'):     # get the clean data from yaml
      input=j.replace('- ','')
      text.append(input)
    f_train.append(text)
  
  
  text=[]
  for i in data['nlu']:
    for j in i['examples'].split('\n'):     # get the clean data from yaml
      input=j.replace('- ','')
      text.append(input)




  raw=[]
  text_clean=[] 
  for x in text:
    if x != '':
      raw.append(x)
      text_clean.append(sbert_model.encode([x])[0])

  X=np.array(text_clean)


  label=[]
  num=[]
  for i in data['nlu']:
    # print(i['examples'].split('\n'))
    num.append(len(i['examples'].split('\n'))-1)
    label.append(i['intent'])
  Y=[]
  label_n=np.arange(len(label))
  for i,j in zip(num,label_n):
    Y.extend(np.ones(i)*j)

  assert len(Y)==X.shape[0]

  s=np.array(Y).astype('int')


  psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
      X, s, clf=LogisticRegression(max_iter=10000, multi_class='auto', solver='lbfgs'))


  return psx,num,raw,s,f_train,label
```

with the input of psx in the built in clean lab package, it returns the indices of misclassifed data
```bash
from cleanlab.pruning import get_noise_indices

 ordered_label_errors = get_noise_indices(
       s=s,
       psx=psx,
       sorted_index_method='prob_given_label',
       prune_method='prune_by_class' # Orders label errors
   )

  ind=np.array(sorted(ordered_label_errors))
```

###  Bert Sentence Embedding 


load a pre_trained bert model for sentence embedding
```bash
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

query = "I had pizza and pasta"
query_vec = sbert_model.encode([query])[0]
```


this function is complete data preprocessing with Berd sentence embedding with the input of yaml files
```bash
def BertEmbed(path):
  import yaml
  with open(path,'r') as f:
    data=yaml.safe_load(f)

  f_train=[]
  for i in data['nlu']:
    text=[]
    for j in i['examples'].split('\n'):     # get the clean data from yaml
      input=j.replace('- ','')
      text.append(input)
    f_train.append(text)
  
  
  text=[]
  for i in data['nlu']:
    for j in i['examples'].split('\n'):     # get the clean data from yaml
      input=j.replace('- ','')
      text.append(input)


  raw=[]
  text_clean=[] 
  for x in text:
    if x != '':
      raw.append(x)
      text_clean.append(sbert_model.encode([x])[0])

  X=np.array(text_clean)

  label=[]
  num=[]
  for i in data['nlu']:
    # print(i['examples'].split('\n'))
    num.append(len(i['examples'].split('\n'))-1)
    label.append(i['intent'])

  Y=[]
  label_n=np.arange(len(label))
  for i,j in zip(num,label_n):
    Y.extend(np.ones(i)*j)

  assert len(Y)==X.shape[0]

  s=np.array(Y).astype('int')
  
  return X,s,f_train
```



## RASA train and test framework
+ [Data_augmentaion_and_confient_learning_RASA_evaluation.ipynb](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/Data_augmentaion_and_confient_learning_RASA_evaluation.ipynb)
  +  Evaluation Framework: The whole framework used for evaluation consists of two main parts: training and testing, (all achieved in RASA). Input data is trained into the RASA training framework. Then trained models are evaluated in the RASA testing framework.


![alt text](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/output_result_by_haoyang/figure/Blank%20diagram%20(2).png)
  + Three pipelines(DIET, DIET_pre_trained, Bert)are stored inside conifg files for RASA training framework to access:1.[DIET](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/RASA_asos/ayo-faq/configs/config-DIET.yml) 2.[DIET_pre_trained](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/RASA_asos/ayo-faq/configs/config-DIET-PreTrained.yml) 3.[BERT](https://github.com/andrewzhai/ASOS_project_haoyang/blob/master/RASA_asos/ayo-faq/configs/config-BERT.yml)

#### Using

The performance for T5 data augmentation and Confident learning are both evaluated in the following framework.The training approaches remain the same, the performance of the training model in the test can reflect the quality of the training data, so that we can evaluate the effect of different data augmentation and confident learning strategies in original input data.

#### RASA training:
This is framework for training different NLU classifers in RASA, you can customized your NLU training data. 

```bash
for i in range(1,6):
  !rasa train nlu --nlu [training files without augmented data] --config [config file] --out models --fixed-model-name [trained models location]


for j in range(1,6):
  for i in range(1,11):  
    !rasa train nlu --nlu [training files with different sizes of augmented data] --config [config file] --out models --fixed-model-name [trained models location]
```

#### RASA test:
This is framework for test different NLU classifers in RASA. Of course you can train your own models following the previous methods, but it takes quite a long time for rasa to train and testing those models we need for plotting. Alternatively, I provided some pre-trained and tested result files. 

In the following code: pipe represents the number of pipeline we use, j represents which fold of training file we used, and i reporesents the number of augmented sentences for each input sentence.

```bash
for pipe in range(1,4):
  for j in range(1,6):
    for i in range(0,11):
      !rasa test nlu --model [pre-trained model] --nlu [test file, no augumented data] --out [result files]
```
The RASA test provides multiple indicators to measure the performance of the training model. During every trained model evaluation in RASA Framework, it returns an intent confusion matrix, an intent_error json file, an intent histogram, and an intent report json.


# Results
The visaulaization plots based on the results generated by RASA test are all stored inside [figure](https://github.com/andrewzhai/ASOS_project_haoyang/tree/master/output_result_by_haoyang/figure)


# Acknowledgements :

Thanks for [Cornor](https://github.com/c-mccabe), and [Fabon](https://github.com/fabon), for lots of advice and instructions in data generation RASA training and testing.
Also thanks for them providing of Augmented data with GPT3 model and demonstraing the research direction.
Thanks for [Hardy](https://github.com/hardyflav) for the advice in Confident learning and PU learning.




