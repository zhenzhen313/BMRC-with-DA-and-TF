from transformers import BertModel
import torch.nn as nn
import torch


class BERTModel(nn.Module):
    def __init__(self, hidden_size, bert_model_type):

        super(BERTModel, self).__init__()

        # BERT模型
        if bert_model_type == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained(bert_model_type)
            print('bert-base-uncased model loaded')
        else:
            raise KeyError('bert_model_type should be bert-based-uncased.')

        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,
                          hidden_size=hidden_size)

        self.classifier_a = nn.Linear(hidden_size, 4)
        self.classifier_ao = nn.Linear(hidden_size, 4)
        self.classifier_o = nn.Linear(hidden_size, 4)
        self.classifier_oa = nn.Linear(hidden_size, 4)
        self.classifier_sentiment1 = nn.Linear(2 * hidden_size, 4)
        self.classifier_sentiment2 = nn.Linear(512, 3)

    def forward(self, query_tensor, query_mask, query_seg, step):

        bert_output = self.bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        classifier_input, _ = self.lstm(bert_output)
        if step == 'A':
            predict = self.classifier_a(classifier_input)
            return predict
        elif step == 'O':
            predict = self.classifier_o(classifier_input)
            return predict
        elif step == 'AO':
            predict = self.classifier_ao(classifier_input)
            return predict
        elif step == 'OA':
            predict = self.classifier_oa(classifier_input)
            return predict
        elif step == 'S':
            classifier_input2, _ = self.gru(bert_output)
            classifier_input = torch.cat((classifier_input * 0.5, classifier_input2 * 0.5), dim=2)
            predict = self.classifier_sentiment1(classifier_input)
            predict = predict.view([predict.size(0), -1])
            predict = self.classifier_sentiment2(predict)
            return predict
        else:
            raise KeyError('step error.')
