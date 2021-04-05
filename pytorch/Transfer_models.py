import torch.nn
from models import *  # import all models


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base_num):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins,
                          fmin, fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        # Freeze AudioSet pretrained layers
        count = 0
        for child in self.base.children():
            if count < freeze_base_num: 
                for param in child.parameters():
                    param.requires_grad = False
                    count += 1

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        #         clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        clipwise_output = torch.sigmoid(self.fc_transfer(embedding))
        output_dict['clipwise_output'] = clipwise_output

        return output_dict


class Transfer_Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base_num):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn10, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn10(sample_rate, window_size, hop_size, mel_bins,
                          fmin, fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)

        # Freeze AudioSet pretrained layers
        count = 0
        for child in self.base.children():
            if count < freeze_base_num: 
                for param in child.parameters():
                    param.requires_grad = False
                    count += 1

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        #         clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        clipwise_output = torch.sigmoid(self.fc_transfer(embedding))
        output_dict['clipwise_output'] = clipwise_output

        return output_dict


class Transfer_Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base_num):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn6, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins,
                          fmin, fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)

        # Freeze AudioSet pretrained layers
        count = 0
        for child in self.base.children():
            if count < freeze_base_num: 
                for param in child.parameters():
                    param.requires_grad = False
                    count += 1

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        #         clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        clipwise_output = torch.sigmoid(self.fc_transfer(embedding))
        output_dict['clipwise_output'] = clipwise_output

        return output_dict
