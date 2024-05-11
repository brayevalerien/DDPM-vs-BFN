import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def append_dims(tensor: torch.Tensor, target_dims: int) -> torch.Tensor:
    assert isinstance(
        target_dims, int), f"Expected 'target_dims' to be an integer, but received {type(target_dims)}."
    tensor_dims = tensor.ndim
    assert tensor_dims <= target_dims, f"Tensor has {tensor_dims} dimensions, but target has {target_dims} dimensions."
    return tensor[(...,) + (None,) * (target_dims - tensor_dims)]


@dataclass
class ContinuousDataLossResult:
    loss: torch.Tensor
    input_data: torch.Tensor
    output_data: torch.Tensor


@dataclass
class DiscreteDataLossResult:
    loss: torch.Tensor
    input_probs: torch.Tensor
    output_probs: torch.Tensor


class BayesianFlow:
    def __init__(
            self,
            num_classes: int = None,
            beta: float = None,
            sigma: float = None,
            reduced_features_binary: bool = False
    ) -> None:
        super(BayesianFlow, self).__init__()
        if reduced_features_binary:
            assert (
                num_classes == 2), f"For `reduced_features_binary` number of classes must be 2, got {num_classes}."
        self.num_classes = num_classes
        self.beta = beta
        self.sigma = sigma
        self.reduced_features_binary = reduced_features_binary

    def get_alpha(self, t):
        return self.beta * t

    def get_beta(self, t):
        return self.beta * (t ** 2.0)

    def get_gamma(self, t):
        return 1 - (self.sigma ** (t * 2.0))

    @staticmethod
    def continuous_output_prediction(
            model: nn.Module,
            mu: torch.Tensor,
            t: torch.Tensor,
            gamma: torch.Tensor,
            t_min: float = 1e-10,
            x_min: float = -1.0,
            x_max: float = 1.0,
            **model_kwargs
    ) -> torch.Tensor:
        output = model(mu, t, **model_kwargs)

        gamma = append_dims(gamma, mu.ndim)
        x_hat = (mu / gamma) - torch.sqrt((1 - gamma) / gamma) * output
        x_hat = torch.clip(x_hat, x_min, x_max)

        condition = t < t_min
        return torch.where(append_dims(condition, x_hat.ndim), torch.zeros_like(x_hat), x_hat)

    def continuous_data_continuous_loss(
            self,
            model: nn.Module,
            target: torch.Tensor,
            reduction: str = 'mean',
            **model_kwargs
    ) -> ContinuousDataLossResult:
        assert self.sigma is not None, "Sigma must be set at initialisation for continuous data."

        bsz = target.shape[0]

        t = torch.rand(bsz, requires_grad=False,
                       device=target.device, dtype=torch.float32)

        gamma = self.get_gamma(t)

        mean = append_dims(gamma, target.ndim) * target
        var = append_dims(gamma * (1 - gamma), target.ndim)
        eps = torch.randn_like(target)
        mu = mean + eps * var.sqrt()

        x_hat = self.continuous_output_prediction(
            model, mu.detach(), t, gamma, **model_kwargs)

        weights = -math.log(self.sigma) / (self.sigma ** (t * 2.0))
        err_sq = ((target - x_hat) ** 2)
        loss_limit_inf = (append_dims(weights, err_sq.ndim) * err_sq)

        if reduction == 'mean':
            loss = loss_limit_inf.mean()
        elif reduction == 'sum':
            loss = loss_limit_inf.sum()
        elif reduction == 'none':
            loss = loss_limit_inf
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

        return ContinuousDataLossResult(
            loss=loss,
            input_data=mu,
            output_data=x_hat
        )

    @torch.inference_mode()
    def continuous_data_sample(
            self,
            model: nn.Module,
            size,
            num_steps: int = 100,
            return_all: bool = False,
            device='cuda',
            **model_kwargs
    ):
        assert self.sigma is not None, "Sigma must be set at initialisation for continuous data."

        outputs_list = []

        mu = torch.zeros(size, device=device)
        rho = 1

        for i in range(1, num_steps + 1):
            t = (i - 1) / num_steps
            t = t * torch.ones((mu.shape[0]), device=mu.device, dtype=mu.dtype)

            gamma = self.get_gamma(t)
            x_hat = self.continuous_output_prediction(
                model, mu, t, gamma, **model_kwargs)

            if return_all:
                outputs_list.append(x_hat)

            alpha = self.sigma ** (-2 * i / num_steps) * \
                (1 - self.sigma ** (2 / num_steps))

            mean = x_hat
            var = torch.full_like(mean, fill_value=(1 / alpha))
            eps = torch.randn_like(x_hat)
            y = mean + eps * var.sqrt()

            mu = ((rho * mu) + (alpha * y)) / (rho + alpha)
            rho = rho + alpha

        t = torch.ones((mu.shape[0]), device=mu.device, dtype=mu.dtype)
        x_hat = self.continuous_output_prediction(
            model, mu, t, self.get_gamma(t), **model_kwargs)

        if return_all:
            outputs_list.append(x_hat)
            return outputs_list
        else:
            return x_hat

    def discrete_output_distribution(
            self,
            model: nn.Module,
            theta: torch.Tensor,
            t: torch.Tensor,
            **model_kwargs
    ) -> torch.Tensor:
        if self.num_classes == 2 and self.reduced_features_binary:
            theta = theta[..., :1]

        output = model(theta, t, **model_kwargs)

        assert output.shape == theta.shape, f"Model output shape {output.shape} does not match input {theta.shape}."

        if self.num_classes == 2 and self.reduced_features_binary:
            p_sub_o_true = torch.sigmoid(output)
            p_sub_o = torch.cat((p_sub_o_true, 1 - p_sub_o_true), dim=-1)
        else:
            p_sub_o = torch.nn.functional.softmax(output, dim=-1)
        return p_sub_o

    def target_to_distribution(self, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.int64:
            target_dist = F.one_hot(
                target, num_classes=self.num_classes).float()
        elif target.dtype in (torch.float16, torch.float32, torch.float64):
            final_dim = target.shape[-1]
            if self.num_classes == 2 and self.reduced_features_binary:
                assert final_dim == 1, \
                    f"Target probabilities final dimension must be 1 for `reduced_features_binary`, got {final_dim}."
                target = torch.cat((target, 1 - target), dim=-1)
            else:
                assert final_dim == self.num_classes, \
                    f"Target probabilities last dimension must match {self.num_classes} classes, got {final_dim}."
            target_dist = target
        else:
            assert False, f"Unsupported dtype {target.dtype}. Supported dtypes are int64 and float types."
        return target_dist

    def discrete_data_continuous_loss(
            self,
            model: nn.Module,
            target: torch.Tensor,
            reduction: str = 'mean',
            **model_kwargs
    ) -> DiscreteDataLossResult:
        assert self.num_classes is not None, "Number of classes must be set at initialisation for discrete data."
        assert self.beta is not None, "Number of classes must be set at initialisation for discrete data."

        bsz = target.shape[0]

        t = torch.rand(bsz, requires_grad=False,
                       device=target.device, dtype=torch.float32)

        target_dist = self.target_to_distribution(target)

        beta = self.beta * (t ** 2)
        mean = append_dims(beta, target_dist.ndim) * \
            (self.num_classes * target_dist - 1)
        var = append_dims(beta * self.num_classes, target_dist.ndim)
        eps = torch.randn_like(mean)
        y = mean + eps * var.sqrt()

        theta = F.softmax(y, dim=-1)

        p_0 = self.discrete_output_distribution(
            model, theta.detach(), t, **model_kwargs)

        e_x, e_hat = target_dist, p_0
        weights = self.num_classes * self.get_alpha(t)
        err_sq = ((e_x - e_hat) ** 2).sum(-1)
        loss_limit_inf = (append_dims(weights, err_sq.ndim) * err_sq)

        if reduction == 'mean':
            loss = loss_limit_inf.mean()
        elif reduction == 'sum':
            loss = loss_limit_inf.sum()
        elif reduction == 'none':
            loss = loss_limit_inf
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

        return DiscreteDataLossResult(
            loss=loss,
            input_probs=theta,
            output_probs=p_0
        )

    @torch.inference_mode()
    def discrete_data_sample(
            self,
            model: nn.Module,
            size,
            num_steps: int = 100,
            return_all: bool = False,
            device='cuda',
            **model_kwargs
    ):
        assert self.num_classes is not None, "Number of classes must be set at initialisation for discrete data."
        assert self.beta is not None, "Beta must be set at initialisation for discrete data."

        outputs_list = []

        theta = torch.ones((*size, self.num_classes),
                           device=device) / self.num_classes

        for i in range(1, num_steps + 1):
            t = (i - 1) / num_steps
            t = t * torch.ones((theta.shape[0]),
                               device=theta.device, dtype=theta.dtype)

            k_probs = self.discrete_output_distribution(
                model, theta, t, **model_kwargs)
            k = torch.distributions.Categorical(probs=k_probs).sample()

            if return_all:
                outputs_list.append(k_probs)

            alpha = self.beta * (2 * i - 1) / (num_steps ** 2)

            e_k = F.one_hot(k, num_classes=self.num_classes).float()
            mean = alpha * (self.num_classes * e_k - 1)
            var = torch.full_like(mean, fill_value=(alpha * self.num_classes))
            eps = torch.randn_like(e_k)
            y = mean + eps * var.sqrt()

            theta = F.softmax(y + torch.log(theta + 1e-10), dim=-1)

        k_probs_final = self.discrete_output_distribution(
            model, theta, torch.ones_like(t), **model_kwargs)

        if return_all:
            outputs_list.append(k_probs_final)
            return outputs_list
        else:
            return k_probs_final


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, output_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class DownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(
            input_dim, output_dim, dropout_prob=dropout_prob)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.conv_block(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(UpBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(input_dim, output_dim, 2, 2)
        self.conv_block = ConvBlock(
            2 * output_dim, output_dim, dropout_prob=dropout_prob)

    def forward(self, x, skip_input):
        x = self.up_sample(x)
        x = torch.concat([x, skip_input], dim=1)
        x = self.conv_block(x)
        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        freq = x @ self.weights.unsqueeze(0)
        return torch.cat([append_dims(x, freq.ndim), freq.sin(), freq.cos()], dim=-1)


class UNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, dropout_prob=0.0):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(128),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(129, 512)
        )
        self.cond_emb = nn.Sequential(
            nn.Embedding(11, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 512),
        )
        self.down1 = DownBlock(input_dim, 128, dropout_prob=dropout_prob)
        self.down2 = DownBlock(128, 256, dropout_prob=dropout_prob)
        self.conv_block = ConvBlock(256, 512, dropout_prob=dropout_prob)
        self.up1 = UpBlock(512, 256, dropout_prob=dropout_prob)
        self.up2 = UpBlock(256, 128, dropout_prob=dropout_prob)
        self.act = nn.SiLU()
        self.output = nn.Conv2d(128, output_dim, kernel_size=1)

    def forward(self, x, t, labels=None):
        if labels is None:
            # All labels are set as unconditional
            labels = torch.zeros(
                x.shape[0], dtype=torch.int64, device=x.device)
        elif self.training:
            # Bias labels for conditioning, set some as zero for unconditional training
            no_condition = torch.rand_like(
                labels, device=x.device, dtype=x.dtype) < 0.1
            labels = (labels + 1).masked_fill(no_condition, 0)
        else:
            # Bias labels for conditioning, ID zero is unconditional
            labels = labels + 1

        t, labels = append_dims(t, x.ndim), append_dims(labels, x.ndim - 1)
        emb = self.time_mlp(t) + self.cond_emb(labels)
        scale, shift = torch.chunk(emb.permute(0, 3, 1, 2), 2, dim=1)

        x = x.permute(0, 3, 1, 2)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x = (x * scale) + shift
        x = self.conv_block(x)
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        x = self.act(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        return x
