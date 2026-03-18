import torch, torch.nn as nn, torch.nn.functional as F
 
def farthest_point_sampling(pts, n_samples):
    B,N,_=pts.shape; device=pts.device
    selected=torch.zeros(B,n_samples,dtype=torch.long,device=device)
    dist=torch.full((B,N),1e10,device=device)
    cur=torch.zeros(B,dtype=torch.long,device=device)
    for i in range(n_samples):
        selected[:,i]=cur
        cpt=pts[torch.arange(B),cur,:].unsqueeze(1)
        d=((pts-cpt)**2).sum(-1)
        dist=torch.min(dist,d)
        cur=dist.argmax(-1)
    return selected
 
def ball_query(pts, queries, radius, max_pts):
    B,N,_=pts.shape; _,M,_=queries.shape
    dists=torch.cdist(queries,pts)
    idx=dists.topk(max_pts,dim=-1,largest=False).indices
    mask=(dists.gather(-1,idx)>radius)
    idx[mask]=idx[:,0:1,:].expand_as(idx)[mask]
    return idx
 
class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_pts, radius, max_pts, in_ch, out_chs):
        super().__init__()
        self.n=n_pts; self.r=radius; self.k=max_pts
        layers=[]; c=in_ch
        for oc in out_chs:
            layers+=[nn.Conv2d(c,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU()]
            c=oc
        self.mlp=nn.Sequential(*layers)
    def forward(self,xyz,feats=None):
        idx=farthest_point_sampling(xyz,self.n)
        new_xyz=xyz[torch.arange(xyz.shape[0])[:,None],idx]
        ball_idx=ball_query(xyz,new_xyz,self.r,self.k)
        grouped=xyz[torch.arange(xyz.shape[0])[:,None,None],ball_idx]
        grouped-=new_xyz.unsqueeze(2)
        if feats is not None:
            gf=feats[torch.arange(feats.shape[0])[:,None,None],ball_idx]
            grouped=torch.cat([grouped,gf],dim=-1)
        x=grouped.permute(0,3,1,2)
        x=self.mlp(x).max(dim=-1).values
      return new_xyz,x.permute(0,2,1)
 
class PointNetPP(nn.Module):
    CLASSES=40  # ModelNet40
    def __init__(self):
        super().__init__()
        self.sa1=PointNetSetAbstraction(512,0.2,32,3,[64,64,128])
        self.sa2=PointNetSetAbstraction(128,0.4,64,128+3,[128,128,256])
        self.fc=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.4),
                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.4),
                               nn.Linear(128,self.CLASSES))
    def forward(self,pts):
        xyz1,f1=self.sa1(pts)
        xyz2,f2=self.sa2(xyz1,f1)
        g=f2.max(1).values
        return self.fc(g)
 
model=PointNetPP()
pts=torch.randn(2,1024,3)
out=model(pts)
print(f"Input: {pts.shape} → Output: {out.shape}")
print(f"Predicted class IDs: {out.argmax(-1).tolist()}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
