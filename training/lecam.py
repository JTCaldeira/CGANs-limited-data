"""
Implements the LeCam loss proposed in Tseng et al. (2021):
"Regularizing Generative Adversarial Networks under Limited Data"
https://arxiv.org/abs/2104.03310
"""

import torch
from torch.nn.functional import relu

from torch_utils import misc, persistence


@persistence.persistent_class
class LeCam:

	def __init__(self, device, decay=0.99):
		self.decay = torch.ones([], device=device) * decay
		self.ema_real = torch.zeros([], device=device)
		self.ema_fake = torch.zeros([], device=device)
		# Only apply regularization after a certain number of steps
		self.do_reg = False

	def loss_real(self, d_real):
		d_real = d_real.detach()

		# Update real scores EMA
		d_real_mean = d_real.mean()
		self.ema_real = (1 - self.decay) * d_real.mean() + self.decay * self.ema_real

		if not self.do_reg:
			return 0

		# LeCam loss for real images
		return torch.square(relu(d_real - self.ema_fake)).mean()

	def loss_fake(self, d_fake):
		d_fake = d_fake.detach()

		# Update fake scores EMA
		self.ema_fake = (1 - self.decay) * d_fake.mean() + self.decay * self.ema_fake

		if not self.do_reg:
			return 0

		# LeCam loss for fake images
		return torch.square(relu(self.ema_real - d_fake)).mean()