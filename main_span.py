# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:46:40 2019

@author: zhan
"""

from model import Span_labler, Pred_finder
import torch.optim as optim
import torch
import torch.nn as nn
from utils import to_var, cuda_num
import random
import json
from load_pretrained_embedding import Glove
from utils import make_span_candidates
import spacy
import en_core_web_sm
from pororo import Pororo

random.seed(10)
syntax_flag = True


data_fn = "data/student_file.json"
 
model_fn = "model/model.params_span_"+str(syntax_flag)
pred_model_fn = "model/pred_model.params_span_"+str(syntax_flag)
confidence_score = 0.1
pred_confidence_score = 0.15

hidden_dim = 200
label_size = 4
n_layers = 2
pos_dim = 20
dp_dim = 20

epoches = 1
learning_rate = 0.01
decay_rate = 0.005
batch_size = 20

max_pred_len = 5
max_arg_len = 10
decay_every = 100

def lr_decay(optimizer, epoch, batch_num, total_batch_num, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**(int((epoch*total_batch_num+batch_num)/decay_every)))
    if batch_num % decay_every == 0:
        print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def span_candidates_filter(candidates,max_len):
    new_candidates = []
    for candidate in candidates:
        if candidate[1] - candidate[0] + 1 <= max_len:
            new_candidates.append(candidate)
    #print("new_candidates", new_candidates)
    return new_candidates

def syntax_check(sent_spacy,rel,dic, span):
    parent = [] # 해당 형태소의 위에 있는 단어 인덱스
    word=[]
    add=0

    if [k for k, v in dic.items() if span[1] in v]==[k for k, v in dic.items() if span[0] in v]:
        return True
    
    for i in range(int(span[0]), int(span[1]+1)):

      word2=""
      for word1, index in dic.items():#해당 형태소를 dic value에서 찾고 word로 바꾸기
   
         if i in index:
          for idx in index:
              if idx==i:
                word2=word1+1
         else:
           continue
         for i in rel:   
                                                                 
             if(i[0]==word2):#해당 단어의 바로 위에 연결되어 있는 노드
              if(i[1]==-1):
                parent.append(i[0]-1)
              else:
                parent.append(i[1]-1)


    for i in range(int(span[0]), int(span[1]+1)):
        if ((dic[parent[i-int(span[0])]][0] >= int(span[0])) and (dic[parent[i-int(span[0])]][-1] <= int(span[1]))):
            pass
        else:
            return False
    return True

def pred_span_candidates_filter(sent_spacy,rel,dic,candidates,max_len):
    candidates = span_candidates_filter(candidates,max_len)
    new_candidates = []
    for candidate in candidates:
        if syntax_check(sent_spacy,rel,dic, candidate):
            new_candidates.append(candidate)
    print("pred_new_candidates", new_candidates)
    print(candidates)
    #return new_candidates
    return candidates            
       
def arg_span_candidates_filter(sent_spacy,rel,dic,candidates,max_len,pred_span):
    candidates = span_candidates_filter(candidates,max_len)
    new_candidates = [pred_span]
    pred_words = [i for i in range(int(pred_span[0]),int(pred_span[1]+1))]
    
    for candidate in candidates:
        add_flag = True
        for word_idx in pred_words:
            if word_idx >= candidate[0] and word_idx <= candidate[1]:
                add_flag = False
                break
        if syntax_check(sent_spacy,rel,dic, candidate) == False:
            add_flag = False
        if add_flag:
            new_candidates.append(candidate)
    return new_candidates

def get_span_head(tokens, rel,dic,candidates, dp2index, dp_idx):
    candidates_head = []
    candidates_head_dp = []
    print(rel)

    for span in candidates:
      #print("span",span)
      word_list=[]
      for word, index in dic.items():
        for i in index:
          if i==span[0]:
            word_list.append(word)
          elif i==span[1]:
            word_list.append(word)
      #print("word",word_list)
      set_word=set(word_list)#중복값제거
      word_list=list(set_word)
      print(word_list)
      save_root=[]
      for i in range(len(word_list)):
        if rel[word_list[i]][1]==-1:
          save_root.append(rel[word_list[i]][0]-1)
          continue
        if rel[word_list[i]][1]-1>=word_list[0] or rel[word_list[i]][1]-1<=word_list[-1]:#word 중에 dep 존재한다면
          save_root.append(rel[word_list[i]][1]-1)


      set_root=set(save_root)#중복값제거
      save_root=list(set_root)
      print("save_root",save_root)
      root=0
      if len(save_root)==1:
        root=save_root[0]
     
      else:
        for i in range(len(save_root)):
          if rel[save_root[i]][1]-1<word_list[0] or rel[save_root[i]][1]-1<word_list[-1]:#root 중복 제거
            if rel[save_root[i]][1]!=-1:
              root=rel[save_root[i]][1]-1
            else:
              root=rel[save_root[i]][0]-1

      span_spacy = tokens[int(span[0]): int(span[1]+1)]
      if(rel[root][1]==-1):
          candidates_head.append(rel[root][0]-1)
      else:
          candidates_head.append(rel[root][1]-1)
      #print("fp_idx",dp_idx)
      #print("dp2indexx",dp2index)
      #if dp_idx[root] in dp2index:
      #candidates_head_dp.append(dp_idx[root])
     # print("dp_idx",dp_idx)
      #print("root",root)
      
      candidates_head_dp.append(dp_idx[dic[root][0]])
    #  else:
     #     candidates_head_dp.append(dp2index["<UNK>"])
    return candidates_head, candidates_head_dp

    
def makeBatch(data,batch_size):
    batch_data = []
    random.shuffle(data)
    batch_num = int(len(data)/batch_size)
    
    for i in range(batch_num):
        batch_data.append(data[i*batch_size:(i+1)*batch_size])
    if len(data) % batch_size != 0:
        batch_data.append(data[(i+1)*batch_size:])
    return batch_data
    
if __name__ == "__main__":
    #nlp = spacy.load("en_core_web_md")
    #nlp = en_core_web_sm.load()
    dp= Pororo(task="dep_parse", lang="ko") #뽀로로 dp
    pos = Pororo(task="pos", lang="ko") # 뽀로로 pos tagging

    with open(data_fn) as f:
        train_data = json.load(f) 
    with open("data/pos2index.json") as f:
        pos2index = json.load(f)
    with open("data/dp2index.json") as f:
        dp2index = json.load(f, strict=False)
    print("load data done")
    glove = Glove("word2vec/kor2 (1).txt")
    model = Span_labler(glove, label_size, len(pos2index), pos_dim, hidden_dim, n_layers, len(dp2index), dp_dim, syntax_flag)
    pred_model = Pred_finder(glove, label_size, len(pos2index), pos_dim, hidden_dim, n_layers)
    if torch.cuda.is_available():
        model = model.cuda(cuda_num)
        pred_model = pred_model.cuda(cuda_num)
    optimizer1 = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(pred_model.parameters(), lr=learning_rate)
    criterion1 = nn.NLLLoss()
    criterion2 = nn.NLLLoss(weight=to_var(torch.FloatTensor([1,5.5])))
    
    for epoch in range(epoches):
        
        batch_data = makeBatch(train_data,batch_size)
        
        for batch_num,batch in enumerate(batch_data):
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            if batch_num % decay_every == 0:
                optimizer1 = lr_decay(optimizer1, epoch, batch_num, len(batch_data),decay_rate, learning_rate)
                optimizer2 = lr_decay(optimizer2, epoch, batch_num, len(batch_data),decay_rate, learning_rate)            
            loss1 = to_var(torch.FloatTensor([0]))
            loss2 = to_var(torch.FloatTensor([0]))
            for instance in batch:
                do_pred = True
               # print("instnace:",instance)
              #  sent_spacy = nlp(instance["sentence"].strip())
                sent_spacy=instance["sentence"];
               ################# 
                print("sent_spacy",sent_spacy);
               
                
                #sent_idx = instance["sentence2index"]
               
                #text = text.strip()
                sent_word = sent_spacy.split()#################공백 기준으로 자른거(단어)
                #print("sent_word",sent_word)
                

                pos_token=pos(sent_spacy)
                dic={}########word와 형태소 딕셔너리로 매핑 인덱스
                i=0
                num=0
                save=0
                sent_pos=[]
                tokens=[]#문장을 형태소 단위로 나눈 토큰
                
                for word in pos_token:
                    if (word[1]=='SPACE'):
                      dic[num]=list(range(save,i))
                      save=i
                      num+=1
                      i-=1
                    else:
                      sent_pos.append(word[1])
                      tokens.append(word[0])
                    i+=1
                sent_idx = [glove.get_word_index(word) for word in tokens]
                #print("tokens",tokens)

                dic[num]=list(range(save,i))
                #if len(sent_spacy) == len(sent_idx):
                print(len(instance["sentence2index"]))
               
                arg_candidates = make_span_candidates(len(instance["sentence2index"]))
                print("arg_cand",arg_candidates)
                pos_idx = []
                dp_idx = []
                #pos_token=pos(sent_spacy)#########포로로로 문장 pos tagging
              #  print(pos2index)
                for token in sent_pos:
              #    print("token",token)
                  if token in pos2index:
                    pos_idx.append(pos2index[token])
                  else:
                        pos_idx.append(pos2index["UNK"])
                rel=[]#############dep_parse 관계 저장한 배열
                dp_token=dp(sent_spacy)#########포로로로 문장 dp, pos 갯수 맞춰서
     
                for token in dp_token:
                  rel.append([token[0],token[2]])###############dep_parse 관계 저장
                  if token[3] in dp2index:
                    for a in range(len(dic[token[0]-1])):
    
                          dp_idx.append(dp2index[token[3]])
                  else:
                    for a in range(len(dic)):
                      for b in dic[a]:
                          dp_idx.append(dp2index["<UNK>"])

              #  for token in sent_spacy:
              #      if token.tag_ in pos2index:
              #          pos_idx.append(pos2index[token.tag_])
              #      else:
              #          pos_idx.append(pos2index["<UNK>"])
              #      if token.dep_ in dp2index:
              #          dp_idx.append(dp2index[token.dep_])
              #      else:
              #          dp_idx.append(dp2index["<UNK>"])
                tuples = instance["tuples"]      

                print("rel",rel)
                print("dic",dic)
                   
                
                word_input = to_var(torch.LongTensor(sent_idx))
                pos_input = to_var(torch.LongTensor(pos_idx))
                dp_input = to_var(torch.LongTensor(dp_idx))
                gold_pred_idx_all = []
                pred_candidates = pred_span_candidates_filter(sent_spacy,rel,dic,arg_candidates,max_pred_len)
                gold_pred_idx = 0
                if len(tuples) == 0:
                    do_pred = False
                for tuple_ in tuples:
                    gold_pred = tuple_["rel_pos"]
                    if gold_pred[-1] == -1:
                        continue
                    if tuple_["score"] > pred_confidence_score:
                        if gold_pred[1] - gold_pred[0] + 1 > max_pred_len or gold_pred not in pred_candidates:
                            do_pred = False
                            continue
                        gold_pred_idx_all.append(pred_candidates.index(gold_pred))
                    else:  
                        do_pred = False
                    if tuple_["score"] > confidence_score:                        
                        arg_candidates = arg_span_candidates_filter(sent_spacy,rel,dic,arg_candidates,max_arg_len,gold_pred)
                        if len(tuple_["arg0_pos"]) == 0:
                            arg0_gold = gold_pred_idx
                        else:
                            if tuple_["arg0_pos"][1] - tuple_["arg0_pos"][0] + 1 > max_arg_len:
                                continue
                            else:
                                if tuple_["arg0_pos"] in arg_candidates:
                                    arg0_gold = arg_candidates.index(tuple_["arg0_pos"])
                                else:
                                    continue
                        
                        args_gold = [gold_pred_idx, gold_pred_idx, gold_pred_idx]
                        go_flag = True
                        for i,arg in enumerate(tuple_["args_pos"]):
                            if i == 3:
                                break
                            else:
                                if arg[1] - arg[0] + 1 > max_arg_len:
                                    go_flag = False
                                    break
                                else:
                                    if arg in arg_candidates:
                                        args_gold[i] = arg_candidates.index(arg)
                                    else:
                                        go_flag = False
                                        break                            
                        if go_flag:
                            candidates_head, candidates_head_dp = get_span_head(tokens, rel,dic, arg_candidates, dp2index, dp_idx)
                            gold_label = to_var(torch.LongTensor([arg0_gold, args_gold[0], args_gold[1], args_gold[2]]))
                            #print(sent_spacy, sent_idx,pos_token, pos_idx, dp_token,dp_idx) 
                            #print("word_input",len(word_input),word_input, "pos_input",len(pos_input),pos_input, "dp_input",len(dp_input),dp_input)
                            arg_out = model(word_input, pos_input, dp_input, arg_candidates, candidates_head, candidates_head_dp, gold_pred_idx, dic)
                            loss1 += tuple_["score"] * criterion1(arg_out.transpose(0,1),gold_label)                        
                
                
                if do_pred:
                    gold_pred = []
                    for i in range(len(pred_candidates)):
                        if i in gold_pred_idx_all:
                            gold_pred.append(1)
                        else:
                            gold_pred.append(0)
                    gold_pred = to_var(torch.LongTensor(gold_pred))
                    pred_out = pred_model(word_input, pos_input, pred_candidates)
                    loss2 += criterion2(pred_out, gold_pred)
            print("model1 ","epoch:",epoch,"batch",batch_num,"loss:",loss1.item()/len(batch))
            print("model2 ","epoch:",epoch,"batch",batch_num,"loss:",loss2.item()/len(batch))
            if loss1.item() != 0:
                loss1.backward()
                optimizer1.step()
            if loss2.item() != 0:
                loss2.backward()
                optimizer2.step()           
            if (batch_num + 1) % 7 == 0:
                torch.save(model.state_dict(), model_fn)
                torch.save(pred_model.state_dict(), pred_model_fn)
        print("epoch:",epoch,'done')
        torch.save(model.state_dict(), model_fn+'_e'+str(epoch))
