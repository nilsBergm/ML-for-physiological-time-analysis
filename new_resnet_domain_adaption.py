import torch.nn as nn
import numpy as np
import torch


    
class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) implementation as a torch autograd.Function.
    GRL is used for domain adaptation by reversing the gradients during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass of the Gradient Reversal Layer.

        Args:
            ctx (torch.autograd.function.Context): Context object to save values for the backward pass.
            x (torch.Tensor): Input tensor.
            alpha (float): Scaling factor for the gradient reversal.

        Returns:
            torch.Tensor: The input tensor, unchanged.
        """
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Gradient Reversal Layer.

        Args:
            ctx (torch.autograd.function.Context): Context object containing saved values from the forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output of the layer.

        Returns:
            torch.Tensor: Reversed gradient multiplied by the scaling factor and applied to the input tensor.
            None: No gradients are computed for the scaling factor.
        """
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainAdaptationResNet1d(nn.Module):
    """
    Domain Adaptation ResNet1d model that combines feature extraction with ResNet1d,
    domain adaptation with a domain classifier, and age prediction with a linear layer.
    """

    def __init__(self, input_dim, blocks_dim, n_classes, n_domains, kernel_size=17, dropout_rate=0.8):
        """
        Initialize the DomainAdaptationResNet1d model.

        Args:
            input_dim (int): Dimensionality of the input data.
            blocks_dim (list): List of tuples specifying the dimensions of the residual blocks in ResNet1d.
            n_classes (int): Number of classes for age prediction.
            n_domains (int): Number of domains for domain adaptation.
            kernel_size (int, optional): Size of the convolutional kernel in ResNet1d. Default is 17.
            dropout_rate (float, optional): Dropout rate for ResNet1d. Default is 0.8.
        """
        super(DomainAdaptationResNet1d, self).__init__()

        # Existing ResNet1d model
        self.resnet1d = ResNet1d(input_dim, blocks_dim, n_classes, kernel_size, dropout_rate)

        # Domain classifier branch
        self.domain_classifier = nn.Sequential(
            nn.Linear(blocks_dim[-1][0] * blocks_dim[-1][1], n_domains),
            nn.LogSoftmax(dim=1)
        )

        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()

    def forward(self, x, domain_label):
        """
        Forward pass of the DomainAdaptationResNet1d model.

        Args:
            x (torch.Tensor): Input tensor.
            domain_label (int): Domain label for domain adaptation.

        Returns:
            torch.Tensor: Predictions for age.
            torch.Tensor: Predictions for domain.
        """
        # Feature extraction with ResNet1d
        features = self.resnet1d(x)

        # Domain adaptation
        alpha = 0.1
        features_reversed = self.gradient_reversal.apply(features,alpha)
        domain_predictions = self.domain_classifier(features_reversed)

        # Age prediction
        age_predictions = self.resnet1d.lin(features)

        return age_predictions, domain_predictions

def _padding(downsample, kernel_size): 
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate): 
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size) 
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False) 
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size) 
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1: 
            maxpool = nn.MaxPool1d(downsample, stride=downsample)#maxpool = nn.MaxPool1d(downsample, stride=downsample, padding = 0)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""

        if self.skip_connection is not None: 
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)


        # 2nd layer
        x = self.conv2(x)
        x += y  
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out) 
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        #self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out) 
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            #resblk1d = ResBlock1d(n_filters_in, n_filters_out, kernel_size = kernel_size, dropout_rate = dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual block
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)


        return x

