1> Class diagram for transformer_scratch

+-------------------+
|   GPTDataset      |
+-------------------+
| - input_ids       |
| - target_ids      |
+-------------------+
| + __init__()      |
| + __len__()       |
| + __getitem__()   |
+-------------------+

+------------------------+
|   MultiHeadAttention   |
+------------------------+
| - d_out                |
| - num_heads            |
| - head_dim             |
| - W_query              |
| - W_key                |
| - W_value              |
| - out_proj             |
| - dropout              |
| - mask                 |
+------------------------+
| + __init__()           |
| + forward()            |
| + multiply()           |
+------------------------+

+-------------------+
|   LayerNorm       |
+-------------------+
| - eps             |
| - scale           |
| - shift           |
+-------------------+
| + __init__()      |
| + forward()       |
+-------------------+

+-------------------+
|   GELU            |
+-------------------+
| + forward()       |
+-------------------+

+-------------------+
|   FeedForward     |
+-------------------+
| - layers          |
+-------------------+
| + __init__()      |
| + forward()       |
+-------------------+

+------------------------+
|   TransformerBlock     |
+------------------------+
| - att                  |
| - ff                   |
| - norm1                |
| - norm2                |
| - drop_shortcut        |
+------------------------+
| + __init__()           |
| + forward()            |
+------------------------+

+-------------------+
|   GPTModel        |
+-------------------+
| - tok_emb         |
| - pos_emb         |
| - drop_emb        |
| - trf_blocks      |
| - final_norm      |
| - out_head        |
+-------------------+
| + __init__()      |
| + forward()       |
+-------------------+

+-------------------+
|   GPTTrainer      |
+-------------------+
| - config          |
| - seed            |
| - device          |
| - model           |
| - tokenizer       |
| - optimizer       |
| - train_loader    |
| - val_loader      |
+-------------------+
| + __init__()      |
| + _setup()        |
| + get_optimizser()|
| + text_to_token_ids()|
| + token_ids_to_text()|
| + load_data()     |
| + create_dataloader()|
| + calc_loss_batch()|
| + calc_loss_loader()|
| + evaluate_model()|
| + generate_and_print_sample()|
| + train_model()   |
| + generate()      |
+-------------------+

2> Interaction diagram

User
 |
 | 1. trainer = GPTTrainer(config)
 v
+-------------------+
|   GPTTrainer      |
+-------------------+
|  __init__()       |
|   |               |
|   |--> GPTModel.__init__()         (creates model)
|   |--> get_optimizser()            (creates optimizer)
|   |--> _setup()                    (sets device, seeds, moves model)
|   |                              
| 2. trainer.load_data()
|   |--> load_data()
|         |--> [downloads/reads text file]
|         |--> create_dataloader(train_data)
|         |     |--> GPTDataset.__init__()
|         |     |--> DataLoader()
|         |--> create_dataloader(val_data)
|         |     |--> GPTDataset.__init__()
|         |     |--> DataLoader()
|
| 3. trainer.train_model()
|   |--> train_model()
|         |--> for each epoch:
|         |      for each batch in train_loader:
|         |         |--> optimizer.zero_grad()
|         |         |--> calc_loss_batch(input, target)
|         |         |     |--> GPTModel.forward()
|         |         |         |--> [TransformerBlock, MultiHeadAttention, etc.]
|         |         |--> loss.backward()
|         |         |--> optimizer.step()
|         |         |--> if eval: evaluate_model()
|         |         |     |--> calc_loss_loader()
|         |         |         |--> calc_loss_batch()
|         |         |             |--> GPTModel.forward()
|         |         |--> generate_and_print_sample()
|         |               |--> text_to_token_ids()
|         |               |--> generate()
|         |                   |--> GPTModel.forward() (repeatedly)
|         |               |--> token_ids_to_text()
|         |               |--> print()
|
|   [returns train_losses, val_losses, tokens_seen]


## sequence diagram

User
 |
 | 1. trainer = Trainer(config)
 v
+-------------------+
|     Trainer       |
+-------------------+
|  __init__()       |
|   |--> Model.__init__()         (creates neural network)
|   |--> sets loss_fn, optimizer
|
 | 2. trainer._setup()
 |   |--> model.to(device)
 |
 | 3. trainer.load_data()
 |   |--> [generates x_data, y_data]
 |   |--> SimpleDataset.__init__()
 |   |--> DataLoader(SimpleDataset)
 |
 | 4. trainer.train_model()
 |   |--> for each epoch:
 |   |      for each batch in train_loader:
 |   |         |--> calc_loss_batch(inputs, targets)
 |   |         |     |--> model(inputs)
 |   |         |         |--> Model.forward()
 |   |         |             |--> layer1, activation, layer2
 |   |         |     |--> loss_fn(outputs, targets)
 |   |         |--> optimizer.zero_grad()
 |   |         |--> loss.backward()
 |   |         |--> optimizer.step()
 |   |--> print loss every 100 epochs
 |
 | 5. trainer.predict(test_value)
 |   |--> model.eval()
 |   |--> model(test_input)
 |   |    |--> Model.forward()
 |   |--> return prediction
 |
 | 6. [Visualization]
 |   |--> Generate x_vals, y_true
 |   |--> model(x_vals)
 |   |    |--> Model.forward()
 |   |--> Plot with matplotlib