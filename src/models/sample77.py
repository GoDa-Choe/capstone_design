import torch, chamfer3D.dist_chamfer_3D

chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
