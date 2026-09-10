r"""
Classifier backbone and likelihood estimator for pattern recognition.
Provides logits, predictive probabilities, epistemic certainty metric \kappa(x),
and exact analytical log-likelihood gradients \nabla_x \log p(y|x).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatternClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits

    def get_confidence(self, x):
        r"""
        Computes epistemic confidence metric \kappa(x) \in [0, 1].
        High confidence indicates clear categorical separation.
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        top1, _ = torch.max(probs, dim=1)
        # Normalized entropy certainty: 1 - H(p)/log(C)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_ent = torch.log(torch.tensor(probs.shape[1], dtype=torch.float32, device=x.device))
        certainty = 1.0 - (entropy / max_ent)
        return torch.clamp(certainty, 0.0, 1.0), probs

    def get_likelihood_gradient(self, x, target_classes=None):
        r"""
        Computes the gradient of the predicted log-likelihood with respect to input:
        \nabla_x \log p(y^* | x)
        """
        x_in = x.detach().requires_grad_(True)
        logits = self.forward(x_in)
        log_probs = F.log_softmax(logits, dim=1)
        
        if target_classes is None:
            # Guide towards highest likelihood mode
            target_classes = torch.argmax(logits, dim=1)
            
        selected_log_probs = log_probs.gather(1, target_classes.unsqueeze(1)).squeeze(1)
        grad = torch.autograd.grad(selected_log_probs.sum(), x_in, create_graph=False)[0]
        return grad.detach()
