import torch

def decoding(model, sentence, dm, device, max_length=50):
    model.eval()
    
    # 1. Handle Input Type (String vs Tensor)
    if isinstance(sentence, str):
        tokens = [dm.tokenizer_fr.encode(token) for token in sentence.split()]
    else:
        tokens = sentence.tolist()

    # Add SOS/EOS if needed
    if tokens[0] != dm.tokenizer_fr.bos_token_id:
        tokens.insert(0, dm.tokenizer_fr.bos_token_id)
    if tokens[-1] != dm.tokenizer_fr.eos_token_id:
        tokens.append(dm.tokenizer_fr.eos_token_id)

    # FIX: Use unsqueeze(0) for Batch-First format (Batch=1, Seq_Len)
    sentence_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # Start with SOS token
    outputs = [dm.tokenizer_en.bos_token_id]

    for i in range(max_length):
        # FIX: Use unsqueeze(0) here too
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        # Greedy Search
        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)

        if best_guess == dm.tokenizer_en.eos_token_id:
            break

    translated_sentence = [dm.tokenizer_en.decode(idx) for idx in outputs]
    
    # Return without SOS
    return translated_sentence[1:]