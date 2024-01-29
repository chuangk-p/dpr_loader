import torch
import pyterrier as pt
import transformers
from transformers import(
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, 
    AutoModel, AutoConfig, AutoTokenizer)
from tqdm import tqdm
from more_itertools import chunked
from pyterrier_dr import BiEncoder
import numpy as np

class TiedmDPR(BiEncoder):
    def __init__(self, model_name, tokenizer_path=None,
                 batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        
        if tokenizer_path is None:
            tokenizer_path = model_name
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # revise
        self.config = AutoConfig.from_pretrained(model_name, use_auth_token=None)
        self.config_dict = vars(self.config)

        self.model =  DPRQuestionEncoder(config=transformers.DPRConfig(**self.config_dict))
        self.model.base_model.bert_model = AutoModel.from_pretrained(
                model_name, use_auth_token=None, **vars(self.config)
            )
        self.model = self.model.to(self.device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path)        
        self.model_context =  DPRContextEncoder(config=transformers.DPRConfig(**self.config_dict))
        self.model_context.base_model.bert_model = AutoModel.from_pretrained(
                model_name, use_auth_token=None, **vars(self.config)
            )
        self.model_context = self.model_context.to(self.device).eval()
        
        self.tokenizer_context = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=100)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).pooler_output
                # res = res[:, 1:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer_context(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model_context(**inps).pooler_output
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)
        
    def getvec(self, sent):
        input_ids = self.tokenizer(sent, return_tensors="pt")["input_ids"].to(self.device)
        embeddings = self.model(input_ids).pooler_output
        
        return embeddings
        
    def sim(self, emb1, emb2):
        return self.cos(emb1, emb2)
        
    def simtext(self, sent1, sent2):
        input_ids = self.tokenizer_context(sent1, return_tensors="pt")["input_ids"].to(self.device)
        embeddings1 = self.model_context(input_ids).pooler_output
        input_ids = self.tokenizer(sent2, return_tensors="pt")["input_ids"].to(self.device)
        embeddings2 = self.model(input_ids).pooler_output
        return self.sim(embeddings1, embeddings2).item()
        
    def simlist(self, ref, sentlst, order=False):
        res = []
        input_ids = self.tokenizer_context(ref, return_tensors="pt")["input_ids"].to(self.device)
        embeddings1 = self.model_context(input_ids).pooler_output
        for sent in sentlst:
          input_ids = self.tokenizer(sent, return_tensors="pt")["input_ids"].to(self.device)
          embeddings2 = self.model(input_ids).pooler_output
          res.append(self.sim(embeddings1, embeddings2).item())

        output = dict(zip(sentlst, res))
        if order == True:
            return dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
        return output
        
    def __repr__(self):
        return f'mDPR({repr(self.model_name)})'