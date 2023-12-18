import torch
import matplotlib.pyplot as plt


def visualize_digit_embedding(digit_embed, save_path, name="base"):
    cossim_mat = []
    for i in range(10):
        cossim = torch.cosine_similarity(digit_embed, digit_embed[i : i + 1, :]).cpu()
        cossim_mat.append(cossim)
    cossim_mat = torch.stack(cossim_mat)
    cossim_mat_nodiag = cossim_mat + torch.diag_embed(torch.nan * torch.ones(10))

    plt.imshow(cossim_mat_nodiag)
    plt.savefig(save_path / "{}.png".format(name))
    return cossim_mat
