
"""
count number of model parameters
"""
def count_params(model):
    num_params = f"number of model parameters: {sum(p.numel() for p in model.parameters())}\n" \
          + f"number of trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n" \
          + f"number of CNN parameters: {sum(p.numel() for p in model.cnn.parameters())}\n" \
          + f"number of trainable CNN parameters: {sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)}\n" \
          + f"number of MLP parameters: {sum(p.numel() for p in model.clf.parameters())}\n" \
          + f"number of trainable MLP parameters: {sum(p.numel() for p in model.clf.parameters() if p.requires_grad)}\n"
    if model.attn:
        num_params += f"number of Attention parameters: {sum(p.numel() for p in model.attn.parameters())}\n" \
          + f"number of trainable Attention parameters: {sum(p.numel() for p in model.attn.parameters() if p.requires_grad)}\n"
    if model.rnn:
        num_params += f"number of RNN parameters: {sum(p.numel() for p in model.rnn.parameters())}\n" \
                     + f"number of trainable RNN parameters: {sum(p.numel() for p in model.rnn.parameters() if p.requires_grad)}"
    return num_params
