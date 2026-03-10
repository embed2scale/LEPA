from torch.nn.functional import mse_loss, smooth_l1_loss, l1_loss, cosine_similarity
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from src.models.vision_transformer import get_transformed_grid
from src.transforms import random_resize_and_rotate


def dot(x, y):
    """
    Computes the dot product between two tensors.
    :param x: Tensor of shape [B, P, D] # where B is batch size, P is number of patches, D is feature dimension
    :param y: Tensor of shape [B, P, D]
    :return: Tensor of shape [] containing mean dot product
    """
    return torch.sum(x * y, dim=-1).mean()

def cosine(x, y):
    """
    Computes the cosine similarity between two tensors.
    :param x: Tensor of shape [B, P, D] # where B is batch size, P is number of patches, D is feature dimension
    :param y: Tensor of shape [B, P, D]
    :return: Tensor of shape [] containing mean cosine similarity
    """
    return cosine_similarity(x, y, dim=-1).mean()

def interpolate_embeddings(embeddings, augmentation_params, input_size, target_size, patch_size, sample_mode='bilinear', upsample = False):
    angle = augmentation_params['angle']
    scale = augmentation_params['scale']
    tx = augmentation_params['tx']
    ty = augmentation_params['ty']
    has_cls_token = embeddings.size(1) % 2 != 0

    # embedding has shape (B, N, D) where N is number of patches (including cls token if present)
    grid_size = int((input_size // patch_size))
    if has_cls_token:
        cls_token, embeddings = embeddings[:, :1], embeddings[:, 1:]
    embeddings = embeddings.clone().reshape(embeddings.size(0), grid_size, grid_size, embeddings.size(2))

    if upsample:

        embeddings = F.interpolate(embeddings.permute(0, 3, 1, 2), size=(input_size, input_size), mode='nearest').permute(0, 2, 3, 1)
        
        # Apply affine transformation to each element in the batch individually
        transformed = []
        for i in range(embeddings.size(0)):
            emb = embeddings[i].permute(2, 0, 1)  # (D, H, W)
            ang = angle[i].item() if angle.dim() > 1 else angle.item()
            scl = scale[i].item() if scale.dim() > 1 else scale.item()
            t_x = tx[i].item() if tx.dim() > 1 else tx.item()
            t_y = ty[i].item() if ty.dim() > 1 else ty.item()
            emb_t = T.functional.affine(
            emb,
            angle=ang,
            translate=[t_x, t_y],
            scale=scl,
            shear=0,
            interpolation=T.InterpolationMode.NEAREST,
            fill=0
            )
            emb_t = T.functional.center_crop(emb_t, target_size)
            # downsample to target_size//patch_size
            emb_t = F.interpolate(emb_t.unsqueeze(0), size=(target_size//patch_size, target_size//patch_size), mode='nearest').squeeze(0)
            transformed.append(emb_t.permute(1, 2, 0))  # (H, W, D)

        embeddings = torch.stack(transformed, dim=0)

    else:
        grid = get_transformed_grid(
            H = target_size//patch_size,
            W = target_size//patch_size,
            patch_size=patch_size,
            theta_deg=angle,
            scale=scale,
            tx=tx,
            ty=ty,
            absolute_coords=True,
        )  # Shape: (B, H, W, 2)

        # Normalize grid to [-1, 1] for grid_sample
        grid[..., 0] = grid[..., 0] *2 / (input_size//patch_size-1)  # x coordinates
        grid[..., 1] = grid[..., 1] *2 / (input_size//patch_size-1)  # y coordinates

        embeddings = F.grid_sample(
            embeddings.permute(0, 3, 1, 2),  # (B, D, H, W)
            grid,  # (B, H_out, W_out, 2)
            mode=sample_mode,
            # align_corners=True,
            padding_mode='zeros'
        ).permute(0, 2, 3, 1)  # (B, H_out, W_out, D)

    embeddings = embeddings.view(embeddings.size(0), -1, embeddings.size(-1))
    if has_cls_token:
        embeddings = torch.cat((cls_token, embeddings), dim=1)
    
    return embeddings


@torch.no_grad()
def mean_reciprocal_rank(encoder, predictor, base_imgs, patch_size, crop_size, condition_on, device, interpolate_not_predict, n_aug=256, sample_mode='bilinear', upsample=False,pangaea_model=False):
    """
    Computes the mean reciprocal rank (MRR) for a given model and sample.
    For each of n_aug augmentations, computes the embedding, then for each
    prediction, ranks the true embedding among all others by similarity.
    """
    encoder.eval()
    predictor.eval()
    # Compute embeddings for the base images
    masks_enc = [torch.arange(0, base_imgs.size(2)//patch_size * base_imgs.size(3)//patch_size, device=device).unsqueeze(0).repeat(base_imgs.size(0), 1)]
    if pangaea_model:
        base_encodings = encoder({"optical": base_imgs.unsqueeze(2)})[0].permute(0,2,3,1) # Shape: (B, H, W, D)
        base_encodings = base_encodings.reshape(base_imgs.size(0), -1, base_encodings.size(-1))  # Shape: (B, N, D)
    else:
        base_encodings = encoder(base_imgs, masks_enc)  # Shape: (B, N, D)
    masks_pred = [torch.arange(0, crop_size//patch_size * crop_size//patch_size, device=device).unsqueeze(0).repeat(n_aug, 1)]
    masks_enc = [torch.arange(0, base_imgs.size(2)//patch_size * base_imgs.size(3)//patch_size, device=device).unsqueeze(0).repeat(n_aug, 1)]

    ranks = []
    for j in range(base_encodings.shape[0]):
        # print(j)
        # Generate n_aug augmentations of the sample using dataset's augmentation method
        augmented_imgs = []
        augmentations = []
        condition_augmentations = []
        for i in range(n_aug):
            aug_img, aug_param = random_resize_and_rotate(base_imgs[j:j+1],[1, crop_size, crop_size])
            augmented_imgs.append(aug_img)
            augmentations.append(aug_param)
            condition_augmentations.append({k: v for k, v in aug_param.items() if k in condition_on})

        augmented_imgs = torch.cat(augmented_imgs, dim=0)  # Shape: (n_aug, C, H, W)
        augmentations = {k: torch.cat([a[k] for a in augmentations], dim=0).to(device) for k in augmentations[0]}  # Shape: (n_aug, ...)
        condition_augmentations = {k: torch.cat([a[k] for a in condition_augmentations], dim=0).to(device) for k in condition_augmentations[0]}  # Shape: (n_aug, ...)

        # Compute embeddings for all augmented images
        if pangaea_model:
            aug_encodings = encoder({"optical": augmented_imgs.unsqueeze(2)})[0].permute(0,2,3,1) # Shape: (n_aug, H, W, D)
            aug_encodings = aug_encodings.reshape(augmented_imgs.size(0), -1, aug_encodings.size(-1))  # Shape: (n_aug, N, D)
        else:
            aug_encodings = encoder(augmented_imgs)  # Shape: (n_aug, N, D)

        if interpolate_not_predict:
            predictions = interpolate_embeddings(base_encodings[j:j+1].repeat(n_aug,1,1), augmentations,input_size=base_imgs.size(2), sample_mode=sample_mode, upsample=upsample, target_size=crop_size, patch_size=patch_size)
        else:
            predictions = predictor(base_encodings[j:j+1].repeat(n_aug,1,1), masks_enc, masks_pred, conditions=condition_augmentations)  # Shape: (n_aug, N, D)

        if aug_encodings.shape[1] % 2 != 0:
            aug_encodings = aug_encodings[:, 1:]  # Remove CLS token if present
        if predictions.shape[1] % 2 != 0:
            predictions = predictions[:, 1:]  # Remove CLS token if present

        dot_matrices = torch.einsum('ijk,ljk->il', predictions, aug_encodings) / (torch.matmul(torch.norm(predictions, dim=-1), torch.norm(aug_encodings, dim=-1).t())+1e-8)

        img_ranks = []
        for i in range(n_aug):
            # Get the similarity scores for the i-th prediction against all encodings
            sims = dot_matrices[i]  # Shape: (n_aug,)
            # Rank the true encoding (i-th) among all others
            rank = (sims > sims[i]).sum().item()+1  # Rank starts at 1
            img_ranks.append(rank)
        ranks.append(img_ranks)

    ranks = torch.tensor(ranks, dtype=torch.float32)
    reciprocal_ranks = 1.0 / ranks
    mrr = reciprocal_ranks.mean()
    mrr_var = reciprocal_ranks.var()
    print(f"Mean Reciprocal Rank (MRR): {mrr}")
    return mrr, mrr_var