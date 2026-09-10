"""Describe cosine distances between verified all-label mean-|TreeSHAP| profiles."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS
from .artifacts import digest,write_json

def visualize(explanations,output):
    source=Path(explanations)/"mean_absolute_shap.csv"
    frame=pd.read_csv(source,index_col=0)
    values=frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError("Invalid SHAP summary")
    valid=np.linalg.norm(values,axis=1)>0
    excluded=frame.index[~valid].tolist()
    frame=frame.loc[valid]
    if len(frame)<3:raise ValueError("At least three nonzero label profiles are required")
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    distance=cosine_distances(frame.to_numpy())
    np.fill_diagonal(distance,0)
    mds=MDS(n_components=2,dissimilarity="precomputed",random_state=42,n_init=4,normalized_stress="auto")
    coordinates=mds.fit_transform(distance)
    output.mkdir(parents=True)
    pd.DataFrame(distance,index=frame.index,columns=frame.index).to_csv(output/"cosine_distances.csv")
    pd.DataFrame(coordinates,index=frame.index,columns=["x","y"]).to_csv(output/"mds_coordinates.csv")
    fig,ax=plt.subplots(figsize=(10,9))
    plot=ax.imshow(distance,vmin=0,vmax=1,cmap="viridis")
    ax.set_xticks(range(len(frame)),frame.index,rotation=90);ax.set_yticks(range(len(frame)),frame.index)
    ax.set_title("Cosine distance between model feature-importance profiles")
    fig.colorbar(plot,ax=ax,label="Cosine distance");fig.tight_layout();fig.savefig(output/"distance_heatmap.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,8))
    ax.scatter(coordinates[:,0],coordinates[:,1])
    for label,(x,y) in zip(frame.index,coordinates):ax.annotate(label,(x,y),fontsize=8)
    ax.set_title("2D MDS of feature-importance distances (approximation)")
    fig.tight_layout();fig.savefig(output/"mds.png",dpi=180);plt.close(fig)
    write_json(output/"visualization_manifest.json",{"input_sha256":digest(source),"stress":float(mds.stress_),
       "excluded_zero_profiles":excluded,"method":"direct cosine distance -> 2D metric MDS; heatmap is the primary distance display",
       "interpretation":"Similarity of model explanations, not measured perceptual similarity or causal proof."})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--explanations",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();visualize(a.explanations,a.output)
