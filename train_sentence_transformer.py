from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM, AutoConfig, BertConfig, AutoModelForCausalLM, BertGenerationDecoder
import pandas as pd
import json
import umap
from custom_TSDAE import *
import os
import umap
import matplotlib.pyplot as plt


if __name__=="__main__":

    # Load data
    finetune_data = pd.read_csv("processed_data/clinvar_processed_snvs.csv")
    test_data = pd.read_csv("processed_data/cogvig_data.csv")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Training on ", device)

    # Instantiate model
    model_name = 'zhihan1996/DNA_bert_3'
    tokenizer = AutoTokenizer.from_pretrained(model_name)#.to(device)

    embedding_model = models.Transformer(model_name).to(device)
    print("Embedding model on CUDA: ", next(embedding_model.parameters()).device)

    pooling_model = models.Pooling(768, 'cls')
    # print("Pooling model on CUDA: ", next(pooling_model.parameters()).device)

    model = SentenceTransformer(modules=[embedding_model, pooling_model], device=device)
    print("Model on CUDA: ", next(model.parameters()).device)
    model.tokenizer = tokenizer

    decoder_config = BertConfig.from_json_file("decoder_config.json")

    # Create dataloaders 
    train_dataset = CustomTSDAE_Dataset(finetune_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=lambda x: x, shuffle=True)

    test_dataset = CustomTSDAE_Dataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda x: x, shuffle=False)


    # Create the joint TSDAE model 
    train_loss = CustomDenoisingAutoEncoderLoss(model, decoder_config, model_name, device=device).to(device)
    print("Train loss encoder on CUDA: ", next(train_loss.encoder.parameters()).device)
    print("Train loss decoder on CUDA: ", next(train_loss.decoder.parameters()).device)

    # Setup callbacks to save best model
    dev_evaluator = LossEvaluator(test_dataloader, loss_model=train_loss, log_dir='logs/', name='dev', device=device)

    best_loss = np.inf
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    def best_score_callback(score, epoch, steps):
        global best_loss
        
        if score<best_loss:
            print(f"New best model - Loss {score}")
            best_loss = score
            model.save('outputs/tsdae-model')
        return

    # Fit model to ClinVar data
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=500,
        steps_per_epoch=50,
        evaluator=dev_evaluator,
        weight_decay=1e-5,
        scheduler='constantlr',
        callback=best_score_callback,
        optimizer_params={'lr': 3e-6},
        show_progress_bar=True
    )

    # Load model with best loss
    best_model = SentenceTransformer('outputs/tsdae-model')

    # Embed Cogvic germline variants
    embedding_results = best_model(tokenizer(test_dataset.mutated_sequences, return_tensors="pt"))
    variants_embeddings = embedding_results["sentence_embedding"].detach().numpy()
    germline_embeddings = test_data[["dbSNP", "Gene", "disease"]].copy()
    germline_embeddings.loc[:, [f"z{i}" for i in range(variants_embeddings.shape[-1])]] = variants_embeddings
    germline_embeddings.to_csv("COGVIC_germline_variants_embeddings.csv", index=False)

    print("\n\nTraining and embedding complete!")