import numpy as np

image = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0]
])
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])


# print(f"{image}\nkernel:{kernel}")
out_h = image.shape[0] - kernel.shape[0] + 1
out_w = image.shape[1] - kernel.shape[1] + 1
out = np.zeros((out_h,out_w))
for i in range(out_h):
    for j in range(out_w):
        patch = image[i:i+kernel.shape[0],
                      j:j+kernel.shape[1]]
        print(f"patch :{patch}")
        out[i,j] = np.sum(kernel * patch)

        print(f"out:{out}")