from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
import torch
import numpy as np


def get_AUROC(labels, measures):
    fpr, tpr, thresholds = roc_curve(labels, measures)
    auroc = auc(fpr, tpr)

    if np.isnan(auroc):
        auroc = 0.0

    return auroc


def get_prediction(train_sample, model):
    preds = []
    for x_test_response in tqdm(train_sample):
        pred = model.predict([x_test_response])
        preds.append(pred[0])

    return preds


def get_prediction_grads(responses, plannings, model):
    preds = []
    for reponse, planning in tqdm(zip(responses, plannings)):
        pred = model.predict(reponse, planning)
        preds.append(pred)

    return preds


def get_gradients(texts, model, tokenizer, embedding_weights):
    """
    Compute per-sample flattened gradients for a batch of texts.
    Returns: [batch_size, num_params] tensor
    """
    model.eval()
    model.zero_grad()

    for p in model.parameters():
        p.requires_grad = False

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Manually perform embedding lookup:
    # Look up the specific embedding vectors for the input IDs
    # Then, we clone() and detach() to create a new TENSOR that is a LEAF node.
    input_text = texts[1]
    target_text = texts[0]

    input_tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    target_tokens = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    
    input_length = input_tokens["input_ids"].shape[1]
    target_length = target_tokens["input_ids"].shape[1]
    
    # Concatenate input + target
    full_text = input_text + target_text
    full_tokens = tokenizer(full_text, return_tensors="pt")
    input_ids = full_tokens["input_ids"].cuda()
    attention_mask = full_tokens.get("attention_mask").cuda() if "attention_mask" in full_tokens else None
    
    # Get embeddings
    initial_embeds = embedding_weights.index_select(dim=0, index=input_ids.flatten()).view(input_ids.shape[0], input_ids.shape[1], -1)
    
    embeds = initial_embeds.clone().detach().requires_grad_(True)
    
    # Create labels: -100 for input (ignored in loss), actual tokens for target
    labels = input_ids.clone()
    labels[:, :input_length] = -100  # Ignore loss on input portion
    
    # Forward pass
    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        output_hidden_states=False
    )
    
    loss = outputs.loss
    loss.backward()

    del outputs
    torch.cuda.empty_cache()

    grad_embed = embeds.grad.detach().cpu()
    grad_mean = grad_embed.mean(dim=1)

    del loss, embeds, grad_embed, initial_embeds
    torch.cuda.empty_cache()

    return grad_mean.tolist()[0]  # [batch_size, num_params]


def get_kernel(embed1, embed2):
    K = torch.tensor(embed1) @ torch.tensor(embed2).T
    
    return K


def get_rbf_kernel(X1, X2, length_scale=10.0, variance=1.0, noise=0.0):
    """
    RBF/Gaussian kernel: k(x,x') = variance * exp(-||x-x'||^2 / (2*length_scale^2))
    """
    X1 = torch.tensor(X1) if isinstance(X1, list) else X1
    X2 = torch.tensor(X2) if isinstance(X2, list) else X2

    X1_norm = (X1 ** 2).sum(1).view(-1, 1)
    X2_norm = (X2 ** 2).sum(1).view(1, -1)
    dists_sq = X1_norm + X2_norm - 2.0 * X1 @ X2.T
    dists_sq = torch.clamp(dists_sq, min=0)
    
    # Use softplus to ensure positive parameters
    if isinstance(length_scale, torch.Tensor):
        length_scale = torch.nn.functional.softplus(length_scale) + 0.1
        dists_sq = dists_sq.to(length_scale.device)
    if isinstance(variance, torch.Tensor):
        variance = torch.nn.functional.softplus(variance) + 0.1
    
    return variance * torch.exp(-dists_sq / (2 * length_scale ** 2))


def get_cholesky(K, noise, jitter):
    # normalize to correlation-like matrix (numerical stability)
    diag_raw = K.diag().clamp_min(1e-12)    # raw diagonal of K before normalization
    d_train_raw = torch.sqrt(diag_raw) # shape [N]

    # normalize to correlation-like matrix (numerical stability)
    K = K / (d_train_raw[:, None] * d_train_raw[None, :] + 1e-12)

    # symmetrize (important)
    K = (K + K.T) / 2.0

    N = K.shape[0]
    # add observation noise on diagonal
    K = K + (noise * torch.eye(N, dtype=K.dtype, device=K.device))

    # compute smallest eigenvalue to decide how much jitter to add
    try:
        eigs = torch.linalg.eigvalsh(K)
    except Exception:
        # fallback if eigvalsh fails for some reason
        eigs = torch.linalg.eigvalsh(K.cpu()).to(K.device)
    min_eig = float(eigs.min().item())

    # If min eigenvalue is negative (or extremely small), add exact jitter to shift it to small positive
    if min_eig <= 0.0:
        required = (-min_eig) + max(jitter, 1e-12)
        K = K + required * torch.eye(N, dtype=K.dtype, device=K.device)
        print(f"[fit] min_eig was {min_eig:.3e}; added required jitter {required:.3e}")

    # Always ensure symmetry after jitter addition
    K = (K + K.T) / 2.0

    # Try Cholesky with an exponential jitter fallback
    attempt_jitter = 0.0
    max_tries = 20
    success = False
    for t in range(max_tries):
        try:
            if attempt_jitter > 0:
                K_try = K + attempt_jitter * torch.eye(N, dtype=K.dtype, device=K.device)
            else:
                K_try = K
            L = torch.linalg.cholesky(K_try)
            success = True
            if attempt_jitter > 0:
                print(f"[fit] Cholesky succeeded after adding extra jitter {attempt_jitter:.3e}")
            break
        except RuntimeError as e:
            # increase attempt_jitter exponentially
            if attempt_jitter == 0:
                attempt_jitter = max(jitter, 1e-12)
            else:
                attempt_jitter = attempt_jitter * 10.0
            print(f"[fit] Cholesky attempt {t} failed; increasing jitter -> {attempt_jitter:.3e}")
    
    if not success:
        # final diagnostics: print min eigenvalue and raise
        try:
            final_min = float(torch.linalg.eigvalsh(K_try).min().item())
        except Exception:
            final_min = None
        raise RuntimeError(f"Cholesky failed after jittering. final_min_eig={final_min}. Last jitter={attempt_jitter}")
    
    return L
