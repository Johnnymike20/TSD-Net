import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
 
 
class BasicConv_FFCA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv_FFCA, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
 
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
 
# class BasicConv_DW(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, dw=False, relu=True, bn=True, bias=False):
#         """
#         - `dw=True`: 使用 DW+PW 结构
#         - `dw=False`: 普通 Conv2d
#         """
#         super(BasicConv_DW, self).__init__()
#         self.out_channels = out_planes

#         if dw and in_planes == out_planes:  # 仅支持输入输出通道相同的情况
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, 
#                           padding=padding, dilation=dilation, groups=in_planes, bias=False),  # Depthwise
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)  # Pointwise
#             )
#         else:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
#                                   padding=padding, dilation=dilation, bias=bias)

#         self.bn = nn.BatchNorm2d(out_planes) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class SEBlock(nn.Module):
#     """ 轻量化 SE 通道注意力模块 """
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         weight = self.pool(x)
#         weight = self.fc(weight)
#         return x * weight


# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.2, map_reduce=8):
#         """
#         1. 增加 SE 轻量注意力机制，提高通道信息利用率
#         2. 适当提升 `scale` 权重，使得 FEM 的贡献更大
#         3. 增强 1x1 Pointwise 卷积，提高信息交互能力
#         """
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce

#         self.branch0 = nn.Sequential(
#             BasicConv_DW(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv_DW(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, dw=True, relu=True),
#             BasicConv_DW(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1, relu=False)  # 额外的 1x1
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv_DW(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv_DW(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1), dw=True),
#             BasicConv_DW((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0), dw=True),
#             BasicConv_DW(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, dw=True, relu=True),
#             BasicConv_DW(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1, relu=False)  # 额外的 1x1
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv_DW(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv_DW(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0), dw=True),
#             BasicConv_DW((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1), dw=True),
#             BasicConv_DW(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, dw=True, relu=True),
#             BasicConv_DW(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1, relu=False)  # 额外的 1x1
#         )

#         self.ConvLinear = BasicConv_DW(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.se = SEBlock(out_planes, reduction=8)  # 增加 SE 注意力
#         self.shortcut = BasicConv_DW(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         out = self.se(out)  # 通过 SE 进一步提升特征表达
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)

#         return out
 
class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv_FFCA(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
 
        self.ConvLinear = BasicConv_FFCA(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv_FFCA(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
 
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
 
        return out

# class SpatialChannelAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatialChannelAttention, self).__init__()
#         # 空间注意力
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         # 通道注意力
#         self.channel_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels//16, 1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels//16, in_channels, 1),
#             nn.Sigmoid()
#         )

# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
        
#         # 添加注意力模块
#         self.attention = SpatialChannelAttention(in_planes)
        
#         # 增加感受野的小分支
#         self.branch0 = nn.Sequential(
#             BasicConv_FFCA(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
#             # 使用更小的卷积核关注小目标
#             BasicConv_FFCA(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, 
#                           padding=1, relu=False)
#         )
        
#         # 增强水平和垂直特征提取
#         self.branch1 = nn.Sequential(
#             BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
#             # 减小步长，增加特征密度
#             BasicConv_FFCA(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), 
#                           stride=1, padding=(0,1)),
#             BasicConv_FFCA((inter_planes//2)*3, 2*inter_planes, kernel_size=(3,1), 
#                           stride=stride, padding=(1,0)),
#             # 多尺度特征提取
#             BasicConv_FFCA(2*inter_planes, 2*inter_planes, kernel_size=3, 
#                           stride=1, padding=3, dilation=3, relu=False)
#         )
        
#         self.branch2 = nn.Sequential(
#             BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv_FFCA(inter_planes, (inter_planes//2)*3, kernel_size=(3,1), 
#                           stride=1, padding=(1,0)),
#             BasicConv_FFCA((inter_planes//2)*3, 2*inter_planes, kernel_size=(1,3), 
#                           stride=stride, padding=(0,1)),
#             # 使用更大的空洞率捕获上下文信息
#             BasicConv_FFCA(2*inter_planes, 2*inter_planes, kernel_size=3, 
#                           stride=1, padding=5, dilation=5, relu=False)
#         )
        
#         # 特征融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(6*inter_planes, 6*inter_planes, 1),
#             nn.BatchNorm2d(6*inter_planes),
#             nn.ReLU(inplace=True)
#         )
        
#         self.ConvLinear = BasicConv_FFCA(6*inter_planes, out_planes, kernel_size=1, 
#                                         stride=1, relu=False)
#         self.shortcut = BasicConv_FFCA(in_planes, out_planes, kernel_size=1, 
#                                       stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         # 应用注意力机制
#         att = self.attention.spatial_att(x) * self.attention.channel_att(x)
#         x = x * att
        
#         # 特征提取
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
        
#         # 特征融合
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.fusion(out)
#         out = self.ConvLinear(out)
        
#         # 残差连接
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)
        
#         return out