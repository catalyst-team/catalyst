from torch import nn
import torchvision
import pretrainedmodels


class ResNetEncoders(nn.Module):
    def __init__(self, encoder_depth, pretrained=False, pool0=False):
        super().__init__()

        if encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
        elif encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        if pool0:
            self.conv1 = nn.Sequential(self.encoder.conv1,
                                       self.encoder.bn1,
                                       self.encoder.relu,
                                       self.encoder.maxpool)
        else:
            self.conv1 = nn.Sequential(self.encoder.conv1,
                                       self.encoder.bn1,
                                       self.encoder.relu)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        return encoder2, encoder3, encoder4, encoder5


class SeResNetEncoders(nn.Module):
    def __init__(self, encoder_depth, pretrained='imagenet', pool0=False):
        super().__init__()

        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained=pretrained)
        else:
            raise NotImplementedError('only 50, 101, 152 version of Resnet are implemented')

        if pool0:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                       self.encoder.layer0.bn1,
                                       self.encoder.layer0.relu1,
                                       self.encoder.layer0.pool0)
        else:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                       self.encoder.layer0.bn1,
                                       self.encoder.layer0.relu1)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        return encoder2, encoder3, encoder4, encoder5


class SeResNetXtEncoders(nn.Module):
    def __init__(self, encoder_depth, pretrained='imagenet', pool0=False):
        super().__init__()

        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=pretrained)
        else:
            raise NotImplementedError('only 50, 101 version of Resnet are implemented')
        if pool0:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                       self.encoder.layer0.bn1,
                                       self.encoder.layer0.relu1,
                                       self.encoder.layer0.pool0)
        else:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                       self.encoder.layer0.bn1,
                                       self.encoder.layer0.relu1)

        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)

        return encoder2, encoder3, encoder4, encoder5


class DenseNetEncoders(nn.Module):
    def __init__(self, encoder_depth, pretrained='imagenet', pool0=False):
        super().__init__()

        if encoder_depth == 121:
            self.encoder = pretrainedmodels.__dict__['densenet121'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 161:
            self.encoder = pretrainedmodels.__dict__['densenet161'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 169:
            self.encoder = pretrainedmodels.__dict__['densenet169'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 201:
            self.encoder = pretrainedmodels.__dict__['densenet201'](num_classes=1000, pretrained=pretrained)
        else:
            raise NotImplementedError('only 121, 161, 169, 201 version of Densenet are implemented')

        if pool0:
            self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                       self.encoder.features.norm0,
                                       self.encoder.features.relu0,
                                       self.encoder.features.pool0)
        else:
            self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                       self.encoder.features.norm0,
                                       self.encoder.features.relu0)

        self.encoder2 = self.encoder.features.denseblock1
        self.transition1 = self.encoder.features.transition1
        self.encoder3 = self.encoder.features.denseblock2
        self.transition2 = self.encoder.features.transition2
        self.encoder4 = self.encoder.features.denseblock3
        self.transition3 = self.encoder.features.transition3
        self.encoder5 = self.encoder.features.denseblock4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        transition1 = self.transition1(encoder2)
        encoder3 = self.encoder3(transition1)
        transition2 = self.transition2(encoder3)
        encoder4 = self.encoder4(transition2)
        transition3 = self.transition3(encoder4)
        encoder5 = self.encoder5(transition3)

        return encoder2, encoder3, encoder4, encoder5