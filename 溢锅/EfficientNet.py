# 导入基础数学库，用于数值计算（如向上取整、乘法等）
import math
# 导入深拷贝模块，用于复制配置参数避免引用传递导致的修改
import copy
# 导入偏函数模块，用于固定函数部分参数（如BatchNorm2d的eps和momentum）
from functools import partial
# 导入有序字典，用于按顺序存储网络层（保证forward顺序与定义一致）
from collections import OrderedDict
# 导入类型注解相关模块，提升代码可读性和类型检查
from typing import Optional, Callable

# 导入PyTorch核心模块，用于构建神经网络
import torch
import torch.nn as nn
from torch import Tensor  # 显式导入Tensor类型注解
from torch.nn import functional as F  # 导入神经网络常用函数（如池化、激活）

# 新增推理所需库：PIL用于图片加载，torchvision.transforms用于图片预处理
from PIL import Image
from torchvision import transforms
# 导入警告过滤模块，屏蔽无关警告（如PIL的DecompressionBombWarning）
import warnings
warnings.filterwarnings("ignore")


# ====================== 工具函数：网络构建基础工具 ======================
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    将通道数调整为8的整数倍（EfficientNet官方设计要求，适配硬件加速）
    :param ch: 原始通道数
    :param divisor: 除数，固定为8（EfficientNet设计规范）
    :param min_ch: 最小通道数，防止调整后通道数过小
    :return: 调整后的通道数（8的整数倍）
    """
    if min_ch is None:
        min_ch = divisor  # 最小通道数默认等于除数（8）
    # 核心逻辑：四舍五入到最近的8的整数倍
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 容错：确保调整后的通道数不小于原始值的90%（避免通道数减少过多）
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    实现Stochastic Depth（随机深度）：按样本维度随机丢弃整个特征图（而非单个元素）
    作用：缓解深层网络过拟合，提升训练稳定性（EfficientNet核心技巧之一）
    :param x: 输入特征图，shape=[B, C, H, W]
    :param drop_prob: 丢弃概率（0~1），0表示不丢弃
    :param training: 是否训练模式（仅训练时生效）
    :return: 处理后的特征图
    """
    if drop_prob == 0. or not training:
        return x  # 不丢弃时直接返回原特征
    keep_prob = 1 - drop_prob
    # 构造与输入维度匹配的随机张量（仅batch维度为B，其余为1，保证每个样本整体丢弃/保留）
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 生成[0,1)随机数 + 保留概率 → 范围[keep_prob, 1+keep_prob)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 向下取整 → 生成0/1张量（1表示保留，0表示丢弃）
    # 特征值除以保留概率（恢复期望强度），再乘以随机张量（实现丢弃）
    output = x.div(keep_prob) * random_tensor
    return output


# ====================== 网络模块：基础组件类 ======================
class DropPath(nn.Module):
    """
    封装随机深度（Stochastic Depth）为可复用的nn.Module模块
    适配PyTorch的nn.Sequential和model.train()/eval()模式
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, x):
        # 调用上述drop_path函数，自动传入training状态（self.training由model.train()/eval()控制）
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    """
    组合模块：卷积(Conv2d) + 批归一化(BN) + 激活函数
    作用：减少代码冗余，统一构建EfficientNet的基础卷积单元
    """
    def __init__(self,
                 input_channel: int,          # 输入通道数
                 output_channel: int,         # 输出通道数
                 kernel_size: int = 3,        # 卷积核大小（默认3x3）
                 stride: int = 1,             # 卷积步长（默认1）
                 groups: int = 1,             # 分组卷积数（1=普通卷积，=input_channel=深度卷积）
                 norm_layer: Optional[Callable[..., nn.Module]] = None,  # BN层类型
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数类型
        # 计算padding：保证卷积后特征图尺寸不变（same padding）
        padding = (kernel_size - 1) // 2
        # 默认BN层为nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认激活函数为SiLU（Swish），EfficientNet官方推荐
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish (torch>=1.7)

        # 调用父类nn.Sequential的构造函数，按顺序添加层
        super(ConvBNActivation, self).__init__(
            # 卷积层：无偏置（BN层已包含偏置，无需重复）
            nn.Conv2d(in_channels=input_channel,
                      out_channels=output_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False),
            # 批归一化层：标准化特征，提升训练稳定性
            norm_layer(output_channel),
            # 激活函数层：引入非线性
            activation_layer()
        )


class SqueezeExcitation(nn.Module):
    """
    SE模块（注意力机制）：通道注意力，提升有效特征的权重
    核心逻辑：压缩(Squeeze)→激励(Excitation)→加权(Scale)
    """
    def __init__(self,
                 input_channel: int,   # MBConv模块的输入通道数
                 expand_channel: int,  # MBConv扩展后的通道数（1x1卷积升维后）
                 squeeze_factor: int = 4):  # 压缩因子（默认4，即通道数缩小4倍）
        super(SqueezeExcitation, self).__init__()
        # 压缩后的通道数 = 输入通道数 / 压缩因子
        squeeze_c = input_channel // squeeze_factor
        # 第一层1x1卷积：压缩通道（降维）
        self.fc1 = nn.Conv2d(expand_channel, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # 激活函数（保持非线性）
        # 第二层1x1卷积：恢复通道（升维）
        self.fc2 = nn.Conv2d(squeeze_c, expand_channel, 1)
        self.ac2 = nn.Sigmoid()  # 激活函数（输出0~1权重）

    def forward(self, x: Tensor) -> Tensor:
        # Step1: Squeeze（压缩）：全局平均池化 → shape=[B, C, 1, 1]
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # Step2: Excitation（激励）：降维→激活→升维→激活（生成通道权重）
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        # Step3: Scale（加权）：权重与原特征逐通道相乘
        return scale * x


class InvertedResidualConfig:
    """
    MBConv模块的配置类：存储单个MBConv块的超参数
    作用：统一管理MBConv的卷积核、通道数、步长、SE模块等配置
    """
    def __init__(self,
                 kernel: int,               # 深度卷积核大小（3或5）
                 input_channel: int,        # 输入通道数
                 out_channel: int,          # 输出通道数
                 expanded_ratio: int,       # 1x1卷积扩展倍数（1或6）
                 stride: int,               # 深度卷积步长（1或2）
                 use_se: bool,              # 是否使用SE模块（EfficientNet均为True）
                 drop_rate: float,          # 随机深度丢弃概率
                 index: str,                # 模块标识（如1a、2b，用于命名）
                 width_coefficient: float): # 宽度倍率因子（控制通道数缩放）
        # 调整输入通道数（按宽度倍率因子缩放，且为8的整数倍）
        self.input_c = self.adjust_channels(input_channel, width_coefficient)
        self.kernel = kernel  # 卷积核大小
        self.expanded_c = self.input_c * expanded_ratio  # 扩展后的通道数
        # 调整输出通道数（按宽度倍率因子缩放，且为8的整数倍）
        self.out_c = self.adjust_channels(out_channel, width_coefficient)
        self.use_se = use_se        # 是否使用SE模块
        self.stride = stride        # 步长
        self.drop_rate = drop_rate  # 随机深度概率
        self.index = index          # 模块标识

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        """静态方法：按宽度倍率因子调整通道数，并保证为8的整数倍"""
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    """
    MBConv模块（Inverted Residual）：EfficientNet的核心构建块
    结构：1x1升维卷积 → 深度可分离卷积 → SE模块 → 1x1降维卷积 → 残差连接
    """
    def __init__(self,
                 cnf: InvertedResidualConfig,  # MBConv配置类实例
                 norm_layer: Callable[..., nn.Module]):  # BN层类型
        super(InvertedResidual, self).__init__()

        # 校验步长合法性（仅支持1或2）
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 判断是否使用残差连接：步长=1 且 输入通道数=输出通道数
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        # 有序字典存储模块层（保证forward顺序）
        layers = OrderedDict()
        activation_layer = nn.SiLU  # 激活函数固定为SiLU

        # Step1: 1x1升维卷积（仅当扩展倍数≠1时添加，避免冗余）
        if cnf.expanded_c != cnf.input_c:
            layers.update({
                "expand_conv": ConvBNActivation(
                    cnf.input_c, cnf.expanded_c,
                    kernel_size=1,  # 1x1卷积
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            })

        # Step2: 深度可分离卷积（分组数=输入通道数 → 深度卷积）
        layers.update({
            "dwconv": ConvBNActivation(
                cnf.expanded_c, cnf.expanded_c,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_c,  # 分组数=通道数 → 深度卷积
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        })

        # Step3: SE模块（可选，EfficientNet均启用）
        if cnf.use_se:
            layers.update({
                "se": SqueezeExcitation(cnf.input_c, cnf.expanded_c)
            })

        # Step4: 1x1降维卷积（激活函数为Identity，无激活）
        layers.update({
            "project_conv": ConvBNActivation(
                cnf.expanded_c, cnf.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity  # 无激活，保证线性降维
            )
        })

        # 组合所有层为Sequential
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c  # 输出通道数（供外部调用）
        self.is_strided = cnf.stride > 1  # 是否是下采样模块

        # Step5: 随机深度（仅当使用残差连接且丢弃概率>0时添加）
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()  # 无操作，保证接口统一

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：MBConv + 残差连接 + 随机深度"""
        result = self.block(x)  # 执行MBConv模块
        result = self.dropout(result)  # 随机深度
        if self.use_res_connect:
            result += x  # 残差连接（输入+输出）
        return result


# ====================== 主网络：EfficientNet核心类 ======================
class EfficientNet(nn.Module):
    """
    EfficientNet主类：整合所有MBConv模块，构建完整网络
    结构：Stem卷积 → 多阶段MBConv → Top卷积 → 全局平均池化 → 分类头
    """
    def __init__(self,
                 width_coefficient: float,   # 宽度倍率因子（控制通道数）
                 depth_coefficient: float,   # 深度倍率因子（控制模块重复次数）
                 num_classes: int = 1000,    # 分类数（默认ImageNet 1000类）
                 dropout_rate: float = 0.2,  # 分类头dropout概率
                 drop_connect_rate: float = 0.2,  # MBConv随机深度概率
                 block: Optional[Callable[..., nn.Module]] = None,  # MBConv模块类
                 norm_layer: Optional[Callable[..., nn.Module]] = None  # BN层类型
                 ):
        super(EfficientNet, self).__init__()

        # 默认MBConv配置（EfficientNet-B0官方配置）
        # 格式：[kernel, in_channel, out_channel, exp_ratio, stride, use_SE, drop_connect_rate, repeats]
        default_cnf = [
            [3, 32, 16, 1, 1, True, drop_connect_rate, 1],  # Stage1: 1个MBConv
            [3, 16, 24, 6, 2, True, drop_connect_rate, 2],  # Stage2: 2个MBConv
            [5, 24, 40, 6, 2, True, drop_connect_rate, 2],  # Stage3: 2个MBConv
            [3, 40, 80, 6, 2, True, drop_connect_rate, 3],  # Stage4: 3个MBConv
            [5, 80, 112, 6, 1, True, drop_connect_rate, 3],  # Stage5: 3个MBConv
            [5, 112, 192, 6, 2, True, drop_connect_rate, 4],  # Stage6: 4个MBConv
            [3, 192, 320, 6, 1, True, drop_connect_rate, 1]   # Stage7: 1个MBConv
        ]

        def round_repeats(repeats):
            """按深度倍率因子调整模块重复次数（向上取整）"""
            return int(math.ceil(depth_coefficient * repeats))

        # 默认MBConv模块为InvertedResidual
        # 把block赋值为InvertedResidual类（不是执行！只是“指向”类）
        if block is None:
            block = InvertedResidual

        # 默认BN层为nn.BatchNorm2d（固定eps=1e-3，momentum=0.1）
        # partial 是 functools 模块提供的偏函数工具，核心作用是：固定一个函数的部分参数，生成一个新的函数，新函数调用时只需传入未被固定的参数，从而简化重复、冗余的参数传递。
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # 偏函数：固定宽度倍率因子，简化通道数调整
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # 偏函数：固定宽度倍率因子，简化MBConv配置创建
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        # 计算所有MBConv模块的总数量（用于分配随机深度概率）
        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []  # 存储所有MBConv的配置

        # 遍历每个Stage的默认配置，生成具体的MBConv配置
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)  # 深拷贝默认配置，避免修改原数据
            # 按深度倍率因子调整当前Stage的模块重复次数
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # 非第一个模块：步长改为1（仅第一个模块下采样），输入通道=输出通道
                    cnf[-3] = 1  # stride
                    cnf[1] = cnf[2]  # input_channel = output_channel

                # 分配随机深度概率（线性递增，保证总概率和为drop_connect_rate）
                cnf[-1] = args[-2] * b / num_blocks
                # 生成模块标识（如1a、2b）
                index = str(stage + 1) + chr(i + 97)  # 97=ord('a')
                # 添加当前MBConv的配置
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # ====================== 构建网络层 ======================
        layers = OrderedDict()

        # Step1: Stem卷积（网络入口，下采样至1/2）
        layers.update({
            "stem_conv": ConvBNActivation(
                input_channel=3,  # 输入为RGB图片，通道数=3
                output_channel=adjust_channels(32),  # 输出通道数（按宽度倍率调整）
                kernel_size=3,
                stride=2,  # 步长=2，下采样
                norm_layer=norm_layer
            )
        })

        # Step2: 构建所有MBConv模块
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # Step3: Top卷积（最后一个1x1卷积，升维至1280）
        last_conv_input_c = inverted_residual_setting[-1].out_c  # 最后一个MBConv的输出通道
        last_conv_output_c = adjust_channels(1280)  # Top卷积输出通道（按宽度倍率调整）
        layers.update({
            "top": ConvBNActivation(
                input_channel=last_conv_input_c,
                output_channel=last_conv_output_c,
                kernel_size=1,  # 1x1卷积
                norm_layer=norm_layer
            )
        })

        # 组合特征提取层（Stem + MBConv + Top）
        self.features = nn.Sequential(layers)
        # Step4: 全局平均池化（将特征图压缩为[B, C, 1, 1]）
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Step5: 分类头（Dropout + 全连接层）
        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))  # Dropout防止过拟合
        # 全连接层：将特征映射到分类数
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # ====================== 初始化权重 ======================
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming正态初始化（适合ReLU/SiLU激活）
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：权重初始化为1，偏置初始化为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 全连接层：正态分布初始化，偏置为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """核心前向传播逻辑（封装为函数，便于复用）"""
        x = self.features(x)  # 特征提取：Stem + MBConv + Top
        x = self.avgpool(x)   # 全局平均池化
        x = torch.flatten(x, 1)  # 展平：[B, C, 1, 1] → [B, C]
        x = self.classifier(x)   # 分类头：输出预测结果
        return x

    def forward(self, x: Tensor) -> Tensor:
        """前向传播入口（调用核心逻辑）"""
        return self._forward_impl(x)


# ====================== 模型构造函数：不同版本的EfficientNet ======================
def efficientnet_b0(num_classes=1000):
    """
    EfficientNet-B0：基础版本
    :param num_classes: 分类数
    :return: EfficientNet-B0模型
    """
    return EfficientNet(width_coefficient=1.0,  # 宽度倍率=1.0
                        depth_coefficient=1.0,  # 深度倍率=1.0
                        dropout_rate=0.2,       # 分类头dropout=0.2
                        num_classes=num_classes)  # 输入尺寸：224x224


def efficientnet_b1(num_classes=1000):
    """EfficientNet-B1：深度倍率提升（1.1），输入尺寸240x240"""
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    """EfficientNet-B2：宽度1.1+深度1.2，输入尺寸260x260"""
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    """EfficientNet-B3：宽度1.2+深度1.4，输入尺寸300x300"""
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    """EfficientNet-B4：宽度1.4+深度1.8，输入尺寸380x380"""
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    """EfficientNet-B5：宽度1.6+深度2.2，输入尺寸456x456"""
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    """EfficientNet-B6：宽度1.8+深度2.6，输入尺寸528x528"""
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    """EfficientNet-B7：宽度2.0+深度3.1，输入尺寸600x600"""
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)


# ====================== 推理模块：图片预处理+模型推理 ======================
def preprocess_image(image_path, input_size=224):
    """
    图片预处理：适配EfficientNet-B0的输入要求（与ImageNet训练时的预处理一致）
    :param image_path: 图片路径（支持绝对/相对路径）
    :param input_size: 输入尺寸（B0默认224，其他版本需对应调整）
    :return: 预处理后的tensor，shape=[1, 3, input_size, input_size]
    """
    # 定义预处理流水线（严格对齐EfficientNet官方训练配置）
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 缩放至输入尺寸
        transforms.ToTensor(),  # 转换为tensor：[H, W, C]→[C, H, W]，值范围[0,255]→[0,1]
        # 归一化：使用ImageNet数据集的均值和方差（EfficientNet预训练权重的标配）
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图片并转换为RGB（避免灰度图/四通道图导致通道数错误）
    image = Image.open(image_path).convert('RGB')
    # 预处理并添加batch维度：[3, H, W]→[1, 3, H, W]（适配模型的batch输入格式）
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def efficientnet_inference(model, image_tensor, device, is_binary=True):
    """
    EfficientNet推理函数：封装推理流程，适配num_classes=1的二分类/回归场景
    :param model: 初始化后的EfficientNet模型
    :param image_tensor: 预处理后的图片tensor
    :param device: 推理设备（cpu/cuda）
    :param is_binary: 是否二分类场景（True=二分类，False=回归）
    :return: 推理结果字典（包含概率/数值、标签、解释）
    """
    # 模型设为评估模式：禁用Dropout、BN层使用滑动平均均值/方差（关键！否则推理结果不稳定）
    model.eval()
    
    # 将模型和数据移到指定设备（保证数据与模型在同一设备）
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    # 推理阶段禁用梯度计算（大幅提升速度，减少内存占用）
    with torch.no_grad():
        output = model.forward(image_tensor)  # 前向传播，输出原始logits
    
    # 结果解析（适配num_classes=1的场景）
    if is_binary:
        # 二分类：sigmoid转换为概率（0~1），0.5为阈值划分正负例
        prob = torch.sigmoid(output).cpu().numpy()[0][0]  # 转CPU+Numpy，取第一个样本的结果
        pred_label = 1 if prob >= 0.5 else 0  # 阈值判断
        return {
            "probability": round(float(prob), 4),  # 保留4位小数
            "pred_label": pred_label,
            "interpretation": "正例" if pred_label == 1 else "负例"  # 结果解释
        }
    else:
        # 回归场景：直接返回预测数值（无激活函数）
        pred_value = output.cpu().numpy()[0][0]
        return {
            "pred_value": round(float(pred_value), 4)
        }


# ====================== 主函数：推理示例 ======================
if __name__ == "__main__":
    # 1. 配置推理参数
    IMAGE_PATH = r"E:\Gong\右锅\SpillOver_20251031_131243_ori\SpillOver_20251031_131243_ori_right_pot1_00000004_frame_monitor_000006.jpg"  # 测试图片路径
    NUM_CLASSES = 1          # 分类数（1=二分类/回归）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备（优先GPU）
    IS_BINARY = True         # True=二分类，False=回归

    # 2. 初始化模型（EfficientNet-B0，分类数=1）
    model = efficientnet_b0(num_classes=NUM_CLASSES)
    # 可选：加载训练好的权重（需保证权重与模型结构匹配）
    # model.load_state_dict(torch.load("efficientnet_b0_weights.pth", map_location=DEVICE))
    print(f"模型初始化完成，推理设备：{DEVICE}")

    # 3. 图片预处理（捕获异常，避免图片加载失败导致程序崩溃）
    try:
        image_tensor = preprocess_image(IMAGE_PATH)
        print(f"图片预处理完成，tensor形状：{image_tensor.shape}")
    except Exception as e:
        print(f"图片加载/预处理失败：{e}")
        exit(1)  # 异常退出

    # 4. 执行推理
    result = efficientnet_inference(model, image_tensor, DEVICE, IS_BINARY)
    
    # 5. 输出推理结果
    print("\n===== 推理结果 =====")
    if IS_BINARY:
        print(f"预测概率：{result['probability']}")
        print(f"预测标签：{result['pred_label']}")
        print(f"结果解释：{result['interpretation']}")
    else:
        print(f"预测数值：{result['pred_value']}")