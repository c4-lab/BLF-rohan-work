import numpy as np
import pandas as pd
import umap
import argparse
import sys

def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    #vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1/np.sqrt(s)))
    return W, -mu

def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)

def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def whiten(vecs,k):
    W, mu = compute_kernel_bias(vecs)
    kernel = W[:,:k]
    return transform_and_normalize(vecs,kernel,mu)

def embed(vecs, do_whiten = True, k = 384, neighbors = 15,rs = None):
    if do_whiten:
        vecs = whiten(vecs,k)
    reducer = umap.UMAP(metric = 'cosine',n_neighbors=neighbors,random_state=rs)
    return reducer.fit_transform(vecs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embed vector space into 2D space using UMAP.')
    parser.add_argument("file",type=str,help="The file to process")
    parser.add_argument("--no-whiten", dest="whiten", action='store_false', default = True,help="Skip whitening the data")
    parser.add_argument("-k",type=int,help="Number of dimensions",default = 300)
    parser.add_argument("-n",type =int, help="Number of nearest neighbors for UMAP",default = 15)

    args = parser.parse_args()
    if args.file.endswith(".feather"):
        df = pd.read_feather(args.file)
    elif args.file.endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        sys.exit(f"Unknown file type: {args.file}")

    drop_cols = ['index','belief','subject','cleansub']
    drop_cols = filter(lambda x: x in df.columns, drop_cols)
    to_embed = df.drop(drop_cols,axis=1).values
    reduced = embed(to_embed,do_whiten =args.whiten,k=args.k,neighbors = args.n)
    fname = args.file.split("/")[-1]
    outfile = f"{fname}{'_whitened_k' if args.whiten else ''}{args.k if args.whiten else ''}_n{args.n}.csv"
    pd.DataFrame(reduced,columns = ["x","y"]).to_csv(outfile,index = False)
    
    