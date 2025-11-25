import time, torch


@torch.no_grad()
def measure_latency_ms(model, batch, n_warm=5, n_iter=20, device='cpu'):
    model.eval()
    batch = batch.to(device)
    for _ in range(n_warm):
        _ = model(batch)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(batch)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    dt = (time.perf_counter()-t0)/n_iter*1000
    return dt



def conv2d_flops(h, w, in_ch, out_ch, k=3, stride=1, padding=1):
    # MACs = H_out*W_out*out_ch*(in_ch*k*k); FLOPs ≈ 2*MACs (mul+add)
    h_out = (h + 2*padding - k)//stride + 1
    w_out = (w + 2*padding - k)//stride + 1
    macs = h_out * w_out * out_ch * (in_ch * k * k)
    return 2 * macs, h_out, w_out

def estimate_flops_tiny_audiocnn(n_mels=64, frames=100, num_classes=2):
    """
    Estimates FLOPs up to each exit for TinyAudioCNN + ExitNet heads.
    Assumes:
      block1: Conv(1->16,k3,p1) + MaxPool(2,2)
      block2: Conv(16->32,k3,p1) + MaxPool(2,2)
      block3: Conv(32->64,k3,p1) + AdaptiveAvgPool(1,1)
      exits: Linear 16->C, 32->C, 64->C
    """
    flops = {}
    total = 0

    # input (1, M, T)
    M, T = n_mels, frames

    # block1 conv
    f1, h1, w1 = conv2d_flops(M, T, 1, 16, k=3, stride=1, padding=1)
    total += f1
    # pool (ignored)
    M1, T1 = h1//2, w1//2
    # exit1 head (Linear 16->C); reduction cost ignored
    flops['exit1'] = total + 2 * (16 * num_classes)

    # block2 conv
    f2, h2, w2 = conv2d_flops(M1, T1, 16, 32, k=3, stride=1, padding=1)
    total += f2
    # pool
    M2, T2 = h2//2, w2//2
    # exit2 head
    flops['exit2'] = total + 2 * (32 * num_classes)

    # block3 conv
    f3, h3, w3 = conv2d_flops(M2, T2, 32, 64, k=3, stride=1, padding=1)
    total += f3
    # GAP ~ negligible
    # final head
    flops['exit3'] = total + 2 * (64 * num_classes)

    return flops
