import torch
import torch.nn as nn
# from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from transformers import BertModel, BertTokenizer, BertConfig

__all__ = ['BertTextEncoder']


# TRANSFORMERS_MAP = {
#     'bert': (BertModel, BertTokenizer),
#     'roberta': (RobertaModel, RobertaTokenizer),
# }

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()

        # tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        # model_class = TRANSFORMERS_MAP[transformers][0]
        # self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        # self.model = model_class.from_pretrained(pretrained)
        # self.use_finetune = use_finetune

        if pretrained == 'bert-base-uncased':
            self.use_finetune = use_finetune
            config = BertConfig.from_json_file("/data/sqhy_model/pretrained/bert-base-uncased/config.json")
            self.model = BertModel.from_pretrained("/data/sqhy_model/pretrained/bert-base-uncased/pytorch_model.bin",
                                                   config=config)
            self.tokenizer = BertTokenizer.from_pretrained('/data/sqhy_model/pretrained/bert-base-uncased/')
        else:
            self.use_finetune = use_finetune
            config = BertConfig.from_json_file("/data/sqhy_model/pretrained/bert-base-chinese/config.json")
            self.model = BertModel.from_pretrained("/data/sqhy_model/pretrained/bert-base-chinese/pytorch_model.bin",
                                                   config=config)
            self.tokenizer = BertTokenizer.from_pretrained('/data/sqhy_model/pretrained/bert-base-chinese/')

    def get_tokenizer(self):
        return self.tokenizer

    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
