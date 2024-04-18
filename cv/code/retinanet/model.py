import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.head import ClassificationHead, RegressionHead
from retinanet.FPN import PyramidFeatureNetwork

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            #out_channel: 64*4(Bottleneck); 64(Basic)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #oup_channel: 128*4 ...
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatureNetwork(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionHead(256)
        self.classificationModel = ClassificationHead(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        ###################################################################
        # TODO: Please substitute the "?" to declare Focal Loss
        ##################################################################

        self.focalLoss = losses.FocalLoss()

        ##################################################################

        self.model_init()

        self.freeze_bn()


    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        Totally the number of blocks of is the parameter with the same name. 
        The first block is a align block with downsample method.
        Then the later block has the same input and output channel with 'planes * expansion'
        '''
        ####################################################################
        # TODO: Please complete the downsample module
        # Hint: Use a "kernel_size=1"'s convolution layer to align the dimension
        #####################################################################
        # downsample = nn.Sequential()
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample.add_module("downsample_conv", nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1
            #                                                    ,padding=0,stride=stride,bias = False))
            # downsample.add_module('dowmsample_bn',nn.BatchNorm2d(planes * block.expansion))

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        ##################################################################

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))     # (planes * block.expansion, planes * block.expansion)

        return nn.Sequential(*layers)
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        ### Backbone
        x = self.conv1(img_batch)   # (B,64,)
        x = self.bn1(x)             
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)         # (B,64*expansion,)
        x2 = self.layer2(x1)        # (B,128*expansion,H/2,W/2)
        x3 = self.layer3(x2)        # (B,256*expansion,H/4,W/4)
        x4 = self.layer4(x3)        # (B,512*expansion,H/8,W/8)
        
        ### Neck
        features = self.fpn([x2, x3, x4])   #x2: c3; x3: c4; x4: c5

        ### Head

        # （batch_size,total_anchor_nums,4）
        # parameter 'total_anchor_nums' depends on 'H,W,feature_size,anchor_num (9)'
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        # （batch_size,total_anchor_nums,class_num）
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        ### Loss Computation / Inference
        anchors = self.anchors(img_batch)
        if self.training:
            return self.forward_train(classification, regression, anchors, annotations)
        else:
            return self.forward_test(classification, regression, anchors, img_batch)

    def forward_train(self, classification, regression, anchors, annotations):
        return self.focalLoss(classification, regression, anchors, annotations)

    def forward_test(self, classification, regression, anchors, img_batch):
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i]) # B, num_anchtors
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


############################################
# resnet layers in each block, for reference

# resnet18: [2, 2, 2, 2]
# resnet34: [3, 4, 6, 3]
# resnet152: [3, 8, 36, 3]
############################################

def resnet50(num_classes, pretrained=False, **kwargs):
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model

def resnet18(num_classes, pretrained=False, **kwargs):
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

def resnet34(num_classes, pretrained=False, **kwargs):
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model

def resnet152(num_classes, pretrained=False, **kwargs):
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model



