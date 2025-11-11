from torch.utils.data import Dataset, DataLoader
import torch
from tokenizers import Tokenizer
from typing import Literal
import pandas as pd
import os


class LanguageTranslation(Dataset):
    def __init__(self, ds:pd.DataFrame, split:Literal['bn', 'en'], src_tokenizer:Tokenizer, tgt_tokenizer:Tokenizer, src_lang:str, tgt_lang:str, seq_len:int):
        
        self.ds = ds
        self.split = split
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor(self.src_tokenizer.token_to_id("[SOS]"))
        self.eos_token = torch.tensor(self.src_tokenizer.token_to_id("[EOS]"))
        self.pad_token = self.src_tokenizer.token_to_id("[PAD]")
        
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_text = self.ds.loc[index, self.src_lang]
        tgt_text = self.ds.loc[index, self.tgt_lang]
        src_encoding = self.src_tokenizer.encode(src_text)
        tgt_encoding = self.tgt_tokenizer.encode(tgt_text)
        
        src_pad_len = self.seq_len - len(src_encoding.ids) - 2
        tgt_pad_len = self.seq_len - len(tgt_encoding.ids) - 1
        
        if src_pad_len < 0 or tgt_pad_len < 0:
            raise ValueError("Seq Length cannot be Less Token Length")
        
        encoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(src_encoding.ids, dtype=torch.int64), 
            self.eos_token, 
            torch.tensor(src_pad_len*self.pad_token, dtype=torch.int64)
        ])
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_encoding.ids, dtype=torch.int64),
            torch.tensor(self.pad_token*tgt_pad_len, dtype=torch.int64)
        ])
        
        label = torch.cat([
            torch.tensor(tgt_encoding.ids, dtype=torch.int64),
            self.eos_token,
            torch.tensor(self.pad_token*tgt_pad_len, dtype=torch.int64)
        ])
        
        
        
        
        
        



