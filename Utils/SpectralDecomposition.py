import os

import numpy as np
import torch

import scipy.sparse.linalg as sp_linalg

def spectral_decomposition(features, adjs, num_view, args):
    UX_features = []
    s_features = []
    U_features = []

    Eigenvalues_adjs = []
    Eigenvectors_adjs = []

    save_dir = "./saved_svd"
    os.makedirs(save_dir, exist_ok=True)

    filenameUX = f"{args.dataset}_UX.pt"
    filenames = f"{args.dataset}_s.pt"
    filenameU = f"{args.dataset}_U.pt"

    filenameEigenvalues = f"{args.dataset}_Eigenvalues.pt"
    filenameEigenvectors = f"{args.dataset}_Eigenvectors.pt"

    file_pathUX = os.path.join(save_dir, filenameUX)
    file_paths = os.path.join(save_dir, filenames)
    file_pathU = os.path.join(save_dir, filenameU)

    file_pathEigenvalues = os.path.join(save_dir, filenameEigenvalues)
    file_pathEigenvectors = os.path.join(save_dir, filenameEigenvectors)
    if os.path.exists(file_paths):
        print("loading saved")
        #
        # UX = torch.load(file_pathUX, map_location='cuda:' + args.device)
        # s = torch.load(file_paths, map_location='cuda:' + args.device)
        # U = torch.load(file_pathU, map_location='cuda:' + args.device)
        #
        # Eigenvalues = torch.load(file_pathEigenvalues, map_location='cuda:' + args.device,)
        # Eigenvectors = torch.load(file_pathEigenvectors, map_location='cuda:' + args.device)
        UX = torch.load(file_pathUX, map_location='cuda:0' )
        s = torch.load(file_paths, map_location='cuda:0' )
        U = torch.load(file_pathU, map_location='cuda:0' )

        Eigenvalues = torch.load(file_pathEigenvalues, map_location='cuda:0' )
        Eigenvectors = torch.load(file_pathEigenvectors, map_location='cuda:0' )

    else:
        for i in range(num_view):

            if isinstance(features[i], np.ndarray):
                UX, s, U = torch.linalg.svd(torch.from_numpy(features[i]))
            else:
                UX, s, U = torch.linalg.svd(features[i].float())

            if hasattr(adjs[i], 'to_dense'):
                temp_adj = adjs[i].to_dense()
            else:
                temp_adj = adjs[i]
            # UX, s, U = torch.linalg.svd(torch.from_numpy(features[i]))
            temp_adj = temp_adj.to(torch.float32)
            temp_adj = (temp_adj + temp_adj.T) / 2
            print("temp_adj dtype:", temp_adj.dtype)
            print("temp_adj shape:", temp_adj.shape)
            print("NaN:", torch.isnan(temp_adj).any().item())
            print("Inf:", torch.isinf(temp_adj).any().item())
            print("Symmetry error:", torch.norm(temp_adj - temp_adj.T))

            # eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(temp_adj))
            eigenvalues, eigenvectors = torch.linalg.eigh(temp_adj)

            Eigenvalues_adjs.append(eigenvalues)
            Eigenvectors_adjs.append(eigenvectors)

            UX_features.append(UX)
            s_features.append(s)
            U_features.append(U)
        # U_feature = F.relu(torch.stack(U_features, dim=1))  # (n, len(ks), nfeat)
        UX = UX_features
        s = s_features
        U = U_features

        Eigenvalues = Eigenvalues_adjs
        Eigenvectors = Eigenvectors_adjs

        torch.save(Eigenvalues, file_pathEigenvalues)
        torch.save(Eigenvectors, file_pathEigenvectors)

        torch.save(UX, file_pathUX)
        torch.save(s, file_paths)
        torch.save(U, file_pathU)

    return Eigenvalues, Eigenvectors, UX, s, U

def spectral_decomposition_large(adjs, num_view, args):


    Eigenvalues_adjs = []
    Eigenvectors_adjs = []

    save_dir = "./saved_svd"
    os.makedirs(save_dir, exist_ok=True)



    filenameEigenvalues = f"{args.dataset}_Eigenvalues.pt"
    filenameEigenvectors = f"{args.dataset}_Eigenvectors.pt"


    file_pathEigenvalues = os.path.join(save_dir, filenameEigenvalues)
    file_pathEigenvectors = os.path.join(save_dir, filenameEigenvectors)
    if os.path.exists(file_pathEigenvalues):
        print("loading saved")
        #
        # UX = torch.load(file_pathUX, map_location='cuda:' + args.device)
        # s = torch.load(file_paths, map_location='cuda:' + args.device)
        # U = torch.load(file_pathU, map_location='cuda:' + args.device)
        #
        # Eigenvalues = torch.load(file_pathEigenvalues, map_location='cuda:' + args.device,)
        # Eigenvectors = torch.load(file_pathEigenvectors, map_location='cuda:' + args.device)


        Eigenvalues = torch.load(file_pathEigenvalues, map_location='cuda:0' )
        Eigenvectors = torch.load(file_pathEigenvectors, map_location='cuda:0' )

    else:
        offset = args.denoise_offset

        for i in range(num_view):

            # if isinstance(features[i], np.ndarray):
            #     UX, s, U = torch.linalg.svd(torch.from_numpy(features[i]))
            # else:
            #     UX, s, U = torch.linalg.svd(features[i].float())

            if hasattr(adjs[i], 'to_dense'):
                temp_adj = adjs[i].to_dense()
                temp_adj = temp_adj.to(torch.float32)
                temp_adj = (temp_adj + temp_adj.T) / 2
                print("temp_adj dtype:", temp_adj.dtype)
                print("temp_adj shape:", temp_adj.shape)
                print("NaN:", torch.isnan(temp_adj).any().item())
                print("Inf:", torch.isinf(temp_adj).any().item())
                print("Symmetry error:", torch.norm(temp_adj - temp_adj.T))
                eigenvalues, eigenvectors = sp_linalg.eigsh(temp_adj.numpy(), k=(args.rewired_index_A + offset))
            else:
                temp_adj = adjs[i]
                eigenvalues, eigenvectors =  sp_linalg.eigsh(temp_adj, k=(args.rewired_index_A+offset))

            # temp_adj = (temp_adj + temp_adj.T) / 2



            # eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(temp_adj))


            Eigenvalues_adjs.append(eigenvalues)
            Eigenvectors_adjs.append(eigenvectors)

        # U_feature = F.relu(torch.stack(U_features, dim=1))  # (n, len(ks), nfeat)

        Eigenvalues = Eigenvalues_adjs
        Eigenvectors = Eigenvectors_adjs

        torch.save(Eigenvalues, file_pathEigenvalues)
        torch.save(Eigenvectors, file_pathEigenvectors)

    print("decomposition done")

    return Eigenvalues, Eigenvectors