# SuperPoint: Self-Supervised Interest Point Detection and Description
## 核心思想
创建一个共同编码器，然后分出“KeyPoint解码器”和“Descriptor解码器”分别输出一个全像素的heatmap以及像素级一一对应的descriptor

## 具体操作
### 编码阶段
作为共同编码器，编码阶段采取两个卷积层，以及三个最大池化层，将原始图像尺寸的(H,W,F)变为(H/8,W/8,F)，其中每个"CELL"代表原图8×8的一个区块。

### KeyPoint解码器  
由于现在编码器输入的为(H/8,W/8)的下采样后的图像，故解码器需要想办法将其重新上采样到原始尺寸，具体操作为：
- 首先使用1×1卷积核对 (H/8,W/8)进行卷积操作，得到(H/8,W/8,65)（第65个通道是对这个8×8区域的定性，如果整个区域都没有KeyPoint，就设置为1，所以又叫做“垃圾通道”）
- 然后对该(H/8,W/8,65)进行上采样，具体操作为直接reshape为(H/8,W/8)的原始图像尺寸，64个通道从左上到右下依次排开，还原为原始尺寸的每个像素对应的输出值，然后进行softmax
  PS：这里之所以1×1卷积能够实现该“左上到右下按序排列”的效果，是监督信号给他的相关提示：“监督信号指出I（1,5）是KeyPoint”这种类似的形式，会使得卷积网络学习到对应的提取模式，再加上Reshape的通道与空间绑定的机制，可以实现还原为原始尺寸的每个像素对应的输出值的效果。

### Descriptor解码器
也是一样，由于现在编码器输入的为(H/8,W/8)的下采样后的图像，故解码器需要想办法将其重新上采样到原始尺寸，具体操作为：
- 首先对(H/8,W/8)进行卷积操作，提取每个CELL的descriptor，形成(H/8,W/8,D)，其中D就是descriptor的维度
- 而后也要进行上采样。这里采取的操作是双三次插值 bicubic interpolation，在(H/8,W/8,D)尺寸下，计算每个原始尺寸在该图像中的位置（例如（8，8）计算对应位置就是（1，1）），而后在该位置取周围的4×4区域的descriptor进行加权平均插值
- 最后要对上采样后的原始图像尺寸的(H,W,D)进行L2 Normalization，以对后面求两个像素的descriptr的余弦相似度来判断是否匹配做铺垫
  PS：这里推理所用的才是上采样后的(H,W,D)，而训练所用的依旧是(H/8,W/8,D)，因为本身descriptor所包含的语义信息就是稀疏的，没必要使用(H,W,D)平白无故增加计算复杂度。

### Loss Funtion
由两部分组成（事实上是三项）：KeyPoint部分 和 Descriptor部分
- KeyPoint部分 有两项组成：（1）原始图像的交叉熵 （2）单应变换之后的交叉熵。
- 其中每个图像的交叉熵计算过程即将每个Cell提取出的logits与标签的独热编码（因为一个Cell只能提取一个KeyPoint）进行交叉熵损失计算
- Descriptor部分为将每个Cell的中心点（这里就对应我们上面提到的训练所用的依旧是(H/8,W/8,D)，每个descriptor事实上都是与一个Cell相对应的）
- 将原图中每个 descriptor 所在 cell 的中心点做单应变换，找到变换图中最邻近的 descriptor；若距离 ≤ 8 px 则判为正样本，否则为负样本。
- 然后对一对正/负样本做余弦相似度度量，之后带入Hinge Loss进行计算

### MagicPoint 和 Homographic Adaptation
- 二者为基础与延申的关系，Homographic Adaptation是本文非常重要的一个创新，同时也是性能提升的非常关键的一个模块
- MagicPoint为在合成几何图形数据集上训练的KeyPoint检测器，但是其在现实生活中的非角形场景中表现并不好，故需要Homographic Adaptation对其进行修正
- Homographic Adaptation核心思想为：当一个图像进行了单应变换后，其变换后对应的KeyPoint进行对应的反变换能够对应上变换前的位置。
- 那么Homographic Adaptation就利用对图像进行Nh次单应变换生成Nh张视图，采用MagicPoint进行关键点检测，而后将这Nh张视图反变换回去，再将这些被反变换的关键点做聚合操作（通常是平均），进而为一张图片做了“伪标签”的生成
- Homographic Adaptation 实现了从“合成数据训练 → 真实图像迁移”的关键跳板，是自监督思想的体现。



