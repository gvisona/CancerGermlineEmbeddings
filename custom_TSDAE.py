import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, InputExample
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import logging
import os
import csv
import numpy as np
from typing import List, Union
import math
from tqdm.autonotebook import trange

from torch.utils.tensorboard import SummaryWriter

from sentence_transformers.evaluation import SentenceEvaluator
# from ..readers import InputExample


logger = logging.getLogger(__name__)

KMER_SIZE = 3

def kmerize_sequence(seq, k_len=3):
    return " ".join([seq[i:i+k_len] for i in range(len(seq)-k_len+1)])

def merge_kmer_sequence(km_seq):
    kmers = km_seq.split(" ")
    return "".join([s[0] for s in kmers[:-1]] + list(kmers[-1]))

class CustomTSDAE_Dataset(Dataset):
    def __init__(self, df, tokenizer, k_size=KMER_SIZE):
        super().__init__()
        print("Preprocessing sequences")
        self.mutated_sequences = [kmerize_sequence(s, k_size) for s in df["MutatedSequence"].values]
        self.reference_sequences = [kmerize_sequence(s, k_size) for s in df["ReferenceSequence"].values]
        self.tokenizer = tokenizer
        print("Preprocessing complete")

    def __len__(self):
        return len(self.mutated_sequences)
        
    def __getitem__(self, idx):
        # return InputExample(str(idx), [self.tokenizer(self.mutated_sequences[idx], return_tensors="pt"), self.tokenizer(self.reference_sequences[idx], return_tensors="pt")], 0)
        return InputExample(str(idx), [self.mutated_sequences[idx], self.reference_sequences[idx]], 0)


class CustomDenoisingAutoEncoderLoss(nn.Module):
    """
        This loss expects as input a batch consisting of damaged sentences and the corresponding original ones.
        The data generation process has already been implemented in readers/DenoisingAutoEncoderReader.py
        During training, the decoder reconstructs the original sentences from the encoded sentence embeddings.
        Here the argument 'decoder_name_or_path' indicates the pretrained model (supported by Huggingface) to be used as the decoder.
        Since decoding process is included, here the decoder should have a class called XXXLMHead (in the context of Huggingface's Transformers).
        Flag 'tie_encoder_decoder' indicates whether to tie the trainable parameters of encoder and decoder,
        which is shown beneficial to model performance while limiting the amount of required memory.
        Only when the encoder and decoder are from the same architecture, can the flag 'tie_encoder_decoder' works.
        For more information, please refer to the TSDAE paper.
    """

    def __init__(
            self,
            model: SentenceTransformer,
            decoder_config,
            decoder_name_or_path,
            tie_encoder_decoder: bool = True,
            device=None
    ):
        """
        :param model: SentenceTransformer model
        :param decoder_config: BertConfiguration for the decoder
        :param tie_encoder_decoder: whether to tie the trainable parameters of encoder and decoder
        """
        super(CustomDenoisingAutoEncoderLoss, self).__init__()
        self.encoder = model  # This will be the final model used during the inference time.
        self.tokenizer_encoder = model.tokenizer
        self._device = device

        encoder_name_or_path = model[0].auto_model.config._name_or_path
        if decoder_config is None:
            assert tie_encoder_decoder, "Must indicate the decoder_name_or_path argument when tie_encoder_decoder=False!"
        if tie_encoder_decoder:
            if decoder_name_or_path:
                logger.warning('When tie_encoder_decoder=True, the decoder_name_or_path will be invalid.')
            decoder_name_or_path = encoder_name_or_path

        self.tokenizer_decoder = self.tokenizer_encoder
        self.need_retokenization = not (type(self.tokenizer_encoder) == type(self.tokenizer_decoder))

        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        kwargs_decoder = {'config': decoder_config}
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path, **kwargs_decoder)
        except ValueError as e:
            logger.error(f'Model name or path "{decoder_name_or_path}" does not support being as a decoder. Please make sure the decoder model has an "XXXLMHead" class.')
            raise e
        assert model[0].auto_model.config.hidden_size == decoder_config.hidden_size, 'Hidden sizes do not match!'
        if self.tokenizer_decoder.pad_token is None:
            # Needed by GPT-2, etc.
            self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        if len(AutoTokenizer.from_pretrained(encoder_name_or_path)) != len(self.tokenizer_encoder):
            logger.warning('WARNING: The vocabulary of the encoder has been changed. One might need to change the decoder vocabulary, too.')

        if tie_encoder_decoder:
            assert not self.need_retokenization, "The tokenizers should be the same when tie_encoder_decoder=True."
            if len(self.tokenizer_encoder) != len(self.tokenizer_decoder):  # The vocabulary has been changed.
                self.tokenizer_decoder = self.tokenizer_encoder
                self.decoder.resize_token_embeddings(len(self.tokenizer_decoder))
                logger.warning('Since the encoder vocabulary has been changed and --tie_encoder_decoder=True, now the new vocabulary has also been used for the decoder.')
            decoder_base_model_prefix = self.decoder.base_model_prefix
            PreTrainedModel._tie_encoder_decoder_weights(
                model[0].auto_model,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix
            )

    def retokenize(self, sentence_features):
        input_ids = sentence_features['input_ids']
        device = input_ids.device
        sentences_decoded = self.tokenizer_encoder.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        retokenized = self.tokenizer_decoder(
            sentences_decoded,
            padding=True,
            truncation='longest_first',
            return_tensors="pt",
            max_length=None
        ).to(device)
        return retokenized

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        source_features, target_features = tuple(sentence_features)
        # for k in source_features:
        #     source_features[k].to(self._device)
        # for k in target_features:
        #     target_features[k].to(self._device)
        # print(target_features)
        if self.need_retokenization:
            # since the sentence_features here are all tokenized by encoder's tokenizer,
            # retokenization by the decoder's one is needed if different tokenizers used
            target_features = self.retokenize(target_features)
        reps = self.encoder(source_features)['sentence_embedding']  # (bsz, hdim)

        # Prepare input and output
        target_length = target_features['input_ids'].shape[1]
        decoder_input_ids = target_features['input_ids'].clone()[:, :target_length - 1]
        label_ids = target_features['input_ids'][:, 1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=source_features['attention_mask'][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        # print(loss)
        return loss


class LossEvaluator(SentenceEvaluator):

    def __init__(self, loader, loss_model: nn.Module = None, name: str = '', log_dir: str = None,
                 show_progress_bar: bool = False, write_csv: bool = True, device=None):

        """
        Evaluate a model based on the loss function.
        The returned score is loss value.
        The results are written in a CSV and Tensorboard logs.
        :param loader: Data loader object
        :param loss_model: loss module object
        :param name: Name for the output
        :param log_dir: path for tensorboard logs 
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """

        self.loader = loader
        self.write_csv = write_csv
        self.logs_writer = SummaryWriter(log_dir=log_dir)
        self.name = name
        self._device = device
        self.loss_model = loss_model
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "loss_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "loss"]

        print("Loss evaluator device: ", self._device)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        self.loss_model.eval()

        loss_value = 0
        self.loader.collate_fn = model.smart_batching_collate
        num_batches = len(self.loader)
        data_iterator = iter(self.loader)

        with torch.no_grad():
            for _ in trange(num_batches, desc="Iteration", smoothing=0.05, disable=not self.show_progress_bar):
                sentence_features, labels = next(data_iterator)
                # print(sentence_features)
                # print(labels)
                for el in sentence_features:
                    # print("---")
                    # print(el)
                    for k in el:
                        el[k] = el[k].to(self._device)
                labels = labels.to(self._device)
                
                # print(sentence_features)
                loss_value += self.loss_model(sentence_features, labels).item()

        final_loss = loss_value / num_batches
        if output_path is not None and self.write_csv:

            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)

            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, final_loss])

            # ...log the running loss
            self.logs_writer.add_scalar('val_loss',
                                        final_loss,
                                        steps)

        self.loss_model.zero_grad()
        self.loss_model.train()

        return final_loss
    