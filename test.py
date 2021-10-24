# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:44:01 2018

@author: win 10
"""

from model import Span_labler,Pred_finder
from load_pretrained_embedding import Glove
import json
import torch
from utils import make_span_candidates, to_var, cuda_num
import spacy
from main_span import pred_span_candidates_filter, arg_span_candidates_filter, get_span_head
from main_span import hidden_dim, label_size, n_layers, pos_dim, dp_dim
import en_core_web_sm
from pororo import Pororo

syntax_flag = True


model_fn = "model/model.params_span_"+str(syntax_flag)+"_e0"
pred_model_fn = "model/pred_model.params_span_True"


def prepare_sentence(sent_spacy,rel,dic,tokens,glove,pos2index,dp2index,pos_idx,dp_idx):
    arg_candidates = make_span_candidates(len(tokens))
    #pred_candidates = pred_span_candidates_filter(sent_spacy, arg_candidates,5)  
    pred_candidates = pred_span_candidates_filter(sent_spacy,rel,dic,arg_candidates,5)
    sent_idx = [glove.get_word_index(word) for word in tokens]
    
    word_input = to_var(torch.LongTensor(sent_idx))
    pos_input = to_var(torch.LongTensor(pos_idx))
    dp_input = to_var(torch.LongTensor(dp_idx))
    return word_input, pos_input, dp_input, pred_candidates, arg_candidates, sent_spacy

def is_overlap(span1,span2): 
    start = [span1[0],span2[0]]
    end = [span1[1],span2[1]]
    if min(end)<max(start):
        return False
    else:
        return True
   
if __name__ == "__main__":
    #nlp = spacy.load("en_core_web_md")
    #nlp = en_core_web_sm.load()
    dp= Pororo(task="dep_parse", lang="ko") #뽀로로 dp
    pos = Pororo(task="pos", lang="ko") # 뽀로로 pos tagging

    glove = Glove("word2vec/kor2 (1).txt")
    with open("data/pos2index.json") as f:
        pos2index = json.load(f)
        pos_size = len(pos2index)
    with open("data/dp2index.json") as f:
        dp2index = json.load(f)
    model = Span_labler(glove, label_size, pos_size, pos_dim, hidden_dim, n_layers, len(dp2index), dp_dim, syntax_flag)
    pred_model = Pred_finder(glove, label_size, pos_size, pos_dim, hidden_dim, n_layers)
    if torch.cuda.is_available():
        model = model.cuda(cuda_num)
        pred_model = pred_model.cuda(cuda_num)
    model.load_state_dict(torch.load(model_fn))
    model.eval()
    pred_model.load_state_dict(torch.load(pred_model_fn))
    pred_model.eval()
 
    sentence = "나는 배가 고파서 집에 가서 빵을 먹었다."
    sent_spacy = sentence

    sent_word = sent_spacy.split()#################공백 기준으로 자른거(단어)

    pos_token=pos(sent_spacy)
    dic={}########word와 형태소 딕셔너리로 매핑 인덱스
    i=0
    num=0
    save=0
    sent_pos=[]
    tokens=[]#문장을 형태소 단위로 나눈 토큰

    for word in pos_token:
        if word[1]=='SPACE':
          dic[num]=list(range(save,i))
          save=i
          num+=1
          i-=1
        else:
          sent_pos.append(word[1])
          tokens.append(word[0])
        i+=1
    dic[num]=list(range(save,i))

    print(dic)
    pos_idx = []
    dp_idx = []

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

                          
    word_input, pos_input, dp_input, pred_candidates, arg_candidates, sent_spacy = prepare_sentence(sent_spacy, rel,dic,tokens,glove, pos2index, dp2index, pos_idx,dp_idx)
    #找出谓词
    pred_out = pred_model(word_input, pos_input, pred_candidates)
    _, max_index = torch.max(pred_out,dim=1)
    gold_pred_all = []
    for i,v in enumerate(max_index):
        if v.data.item() == 1:
            pair = pred_candidates[i]
            gold_pred_all.append(pair)
            
    print("---")
    print("gold",gold_pred_all)
    new_gold_pred_all = []
    mark = [0 for t in gold_pred_all]
    for pred_id, gold_pred in enumerate(gold_pred_all):
        if mark[pred_id] == 0:
            combine_flag = False
            for j in range(len(gold_pred_all)):
                if mark[j] == 0:
                    if gold_pred[1] + 1 == gold_pred_all[j][0]:
                        new_gold_pred_all.append([gold_pred[0], gold_pred_all[j][1]])
                        mark[pred_id] = 1
                        mark[j] = 1
                        combine_flag = True
                        break
                    if gold_pred[0] - 1 == gold_pred_all[j][1]:
                        new_gold_pred_all.append([gold_pred_all[j][0],gold_pred[1]])
                        mark[pred_id] = 1
                        mark[j] = 1
                        combine_flag = True
                        break
            if combine_flag == False and gold_pred not in new_gold_pred_all:
                new_gold_pred_all.append(gold_pred)
                mark[pred_id] = 1
                        
    for gold_pred in new_gold_pred_all:
        do_flag = True
        for pair in new_gold_pred_all:
            if pair != gold_pred:
                if (pair[0] <= gold_pred[0]) and (pair[1] >= gold_pred[1]):
                    do_flag = False
                    print("skip")
                    break
        if do_flag == False:
            continue
        print("gold_red_all",new_gold_pred_all)
        print("gold_pred:",gold_pred)
        print(tokens[gold_pred[0]:gold_pred[1]+1])
        fill = [0,0,0,0]
        fill_span = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
        arg_candidates = arg_span_candidates_filter(sent_spacy,rel,dic,arg_candidates,500,gold_pred)
        candidates_head, candidates_head_dp = get_span_head(tokens, rel,dic, arg_candidates, dp2index, dp_idx)
        gold_pred_idx = arg_candidates.index(gold_pred)
        arg_out = model(word_input, pos_input, dp_input, arg_candidates, candidates_head, candidates_head_dp, gold_pred_idx, dic)
        arg_out = arg_out.transpose(0,1) #(num_label, num_span)
        arg_out = arg_out.cpu().detach().numpy()
        flatten = {}
        for i in range(arg_out.shape[0]):
            for j in range(arg_out.shape[1]):
                flatten[i+10*j] = arg_out[i][j]
        flatten_sorted = sorted(flatten.items(), key=lambda item:item[1], reverse=True)
        for pair in flatten_sorted:
            if fill == [1,1,1,1]:
                break
            else:
                label = int(pair[0] % 10)
                span_idx = int(pair[0] / 10)
                if fill[label] == 1:
                    continue
                elif span_idx == gold_pred_idx:
                    fill[label] = 1
                else:
                    span = arg_candidates[span_idx]
                    add_flag = True
                    

                    for selected_span in fill_span:
                        if is_overlap(span, selected_span):
                            add_flag = False
                            break
                    if label == 0:
                        if span[0] >= gold_pred[0]:
                            add_flag = False
                    if label == 1:
                        if span[0] <= gold_pred[0]:
                            add_flag = False
                    if add_flag:
                        fill[label] = 1
                        fill_span[label] = span
                        print("arg",str(label),": ",sent_spacy[span[0]:span[1]+1])
        print("-----")
            
            
    
