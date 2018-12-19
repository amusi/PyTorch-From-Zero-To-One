# Summary: 检测当前Pytorch和设备是否支持CUDA和cudnn
# Author:  Amusi
# Date:    2018-12-20 
# github:  https://github.com/amusi/PyTorch-From-Zero-To-One

import torch
 
if __name__ == '__main__':
	print("Support CUDA ?: ", torch.cuda.is_available())
	x = torch.Tensor([1.0])
	xx = x.cuda()
	print(xx)
 
	y = torch.randn(2, 3)
	yy = y.cuda()
	print(yy)
 
	zz = xx + yy
	print(zz)
 
	# CUDNN TEST
	from torch.backends import cudnn
	print("Support cudnn ?: ",cudnn.is_acceptable(xx))

