# coding: utf-8
"""
@File        :   train.py
@Time        :   2025/10/21 18:16:19
@Author      :   Usercyk
@Description :   Trains a Physics-Informed Neural Network (PINN) to solve the 3D Poisson equation.
                 Implements the PINN model and training pipeline with distributed training support.
"""
from typing import List, override
from itertools import product

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.optim import Optimizer

from safetensors.torch import save_file

from accelerate import Accelerator

import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange, tqdm


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model for solving
    partial differential equations.

    The network architecture consists of multiple fully-connected
    layers with a configurable activation function.
    """

    def __init__(self, layer_sizes: List[int], activation: nn.Module = nn.Tanh()) -> None:
        """
        Initializes the PINN model with specified layer configuration.

        Args:
            layer_sizes (List[int]):
                Number of neurons in each layer (input to output)
            activation (nn.Module, optional):
                Activation function between linear layers. Defaults to nn.Tanh().
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.activation = activation

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (sample_num, 3) representing 3D coordinates

        Returns:
            Tensor: Output tensor of shape (sample_num, 1) representing the solution
        """
        x = x.contiguous()
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x.contiguous()


class PoissonTrainer:
    """
    Training pipeline for the PINN model to solve the 3D Poisson equation.
    Handles model training, residual calculation, and visualization of results.
    Supports distributed training via Accelerate.
    """

    def __init__(self,
                 layer_sizes: List[int],
                 accelerator: Accelerator,
                 sample_num: int = 1024,
                 learning_rate: float = 0.001,
                 debug: bool = False
                 ) -> None:
        """
        Initializes the Poisson equation trainer.

        Args:
            layer_sizes (List[int]): Network architecture configuration
            accelerator (Accelerator): Distributed training accelerator
            sample_num (int, optional): Number of sample points per batch. Defaults to 1024.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
        """
        self.model = PINN(layer_sizes)
        self.sample_num = sample_num
        self.optimizer: Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.accelerator = accelerator

        self.debug = debug

        self.prepare()

        self.train_result = []

        self.data_path = "/home/stu2400011486/assignments/assignment3/data"

    def get_path(self, file: str) -> str:
        """
        Get the absolute path in the data dir

        Args:
            file (str): The name of the file

        Returns:
            str: The absolute path
        """
        if not self.data_path:
            self.data_path = "/home/stu2400011486/assignments/assignment3/data"
        return f"{self.data_path}/{file}"

    def prepare(self) -> None:
        """
        Prepares the model and optimizer for distributed training.
        Moves model to accelerator device and wraps with distributed training handlers.
        """
        self.model.to(self.accelerator.device)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer)

    def train(self, num_epochs: int) -> None:
        """
        Trains the PINN model for the specified number of epochs.

        Args:
            num_epochs (int): Total number of training epochs
        """
        self.train_result = []
        for epoch in trange(num_epochs,
                            desc="Epoch: ",
                            dynamic_ncols=True,
                            disable=not self.accelerator.is_local_main_process):
            points = self.sample_points(self.sample_num)

            # is_boundary = ((points.abs() >= 0.99).any(dim=1)).float()
            # weights = 1.0 + 4.0 * is_boundary
            residual = self.compute_pde_residual(points)
            loss = torch.mean(residual**2)

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            self.train_result.append((epoch+1, loss.detach().cpu()))
            if (epoch+1) % 100 == 0 and self.accelerator.is_local_main_process:
                tqdm.write(
                    f"Epoch {epoch+1}/{num_epochs} | PDE Loss: {loss.item():.6f}")

        if self.debug and self.accelerator.is_local_main_process:
            # ! UNSAFE
            torch.save(self.model.state_dict(), self.get_path("pinn.pth"))
            # * SAFE
            save_file(self.model.state_dict(),
                      self.get_path("model.safetensors"))

            self.save_train_plot()
            self.plot_phi_slice()
            self.plot_residual_slice()

    def sample_points(self, num_points: int) -> Tensor:
        """
        Generates random sample points in the 3D domain [-1, 1]^3.

        Args:
            num_points (int): Number of points to sample

        Returns:
            Tensor: Sampled points tensor of shape (num_points, 3)
        """
        return torch.rand(num_points, 3, device=self.accelerator.device) * 2 - 1

    def phi(self, coords: Tensor) -> Tensor:
        """
        Computes the solution using the hard constraint formulation:
        φ(x,y,z) = (1-x²)(1-y²)(1-z²)*NN(x,y,z)

        Args:
            coords (Tensor): Input coordinates tensor of shape (sample_num, 3)

        Returns:
            Tensor: Solution values tensor of shape (sample_num, 1)
        """
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        raw_out = self.model(coords)
        return (1 - x ** 2) * (1 - y ** 2) * (1 - z ** 2) * raw_out

    def compute_pde_residual(self, positions: Tensor) -> Tensor:
        """
        Computes the Poisson equation residual: ∇²φ + ρ = 0

        Args:
            positions (Tensor): Positions to evaluate the residual

        Returns:
            Tensor: Residual values tensor of shape (sample_num, 1)
        """
        positions.requires_grad_(True)
        phi = self.phi(positions)

        grads = torch.autograd.grad(
            phi, positions, torch.ones_like(phi), create_graph=True)[0]
        d2phi_dx2 = torch.autograd.grad(grads[:, 0], positions, torch.ones_like(
            grads[:, 0]), create_graph=True)[0][:, 0:1]
        d2phi_dy2 = torch.autograd.grad(grads[:, 1], positions, torch.ones_like(
            grads[:, 1]), create_graph=True)[0][:, 1:2]
        d2phi_dz2 = torch.autograd.grad(grads[:, 2], positions, torch.ones_like(
            grads[:, 2]), create_graph=True)[0][:, 2:3]
        laplacian = d2phi_dx2 + d2phi_dy2 + d2phi_dz2

        x, y, z = positions[:, 0:1], positions[:, 1:2], positions[:, 2:3]
        rho = 100 * x * y * z ** 2

        return laplacian + rho

    def save_train_plot(self, file_name: str = "loss.png") -> None:
        """
        Saves training loss plot as PNG image.

        Args:
            file_name (str, optional): Output filename. Defaults to "loss.png".
        """
        if not self.accelerator.is_local_main_process:
            return

        if not hasattr(self, "train_result") or len(self.train_result) == 0:
            print("No training data found. Please run `train()` first.")
            return

        epochs, losses = zip(
            *[(e, l.item() if torch.is_tensor(l) else l) for e, l in self.train_result])

        plt.figure(figsize=(6, 4))
        plt.scatter(epochs, losses, s=10, color="tab:blue",
                    alpha=0.7, label="PDE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Scatter Plot")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.get_path(file_name), dpi=300)
        plt.close()

    def plot_phi_slice(self, z_value: float = 0.0, num_points: int = 100) -> None:
        """
        Visualizes the solution φ on a 2D slice at constant z.

        Args:
            z_value (float, optional): z-coordinate for the slice. Defaults to 0.0.
            num_points (int, optional): Resolution along x and y axes. Defaults to 100.
        """
        if not self.accelerator.is_local_main_process:
            return

        _x = np.linspace(-1, 1, num_points)
        _y = np.linspace(-1, 1, num_points)
        x, y = np.meshgrid(_x, _y)

        coords = np.stack(
            [x.ravel(), y.ravel(), np.full_like(x.ravel(), z_value)], axis=-1)
        coords_tensor = torch.tensor(
            coords, dtype=torch.float32, device=self.accelerator.device)

        self.model.to(self.accelerator.device)
        with torch.no_grad():
            phi_vals = self.phi(coords_tensor).cpu().numpy()

        phi = phi_vals.reshape(num_points, num_points)

        plt.figure(figsize=(6, 5))
        plt.pcolormesh(x, y, phi, shading='auto', cmap='viridis')
        plt.colorbar(label='Phi')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Phi at z={z_value}')
        plt.tight_layout()
        plt.savefig(self.get_path("phi.png"), dpi=300)
        plt.close()

    def plot_residual_slice(self, z_value: float = 0.0, num_points: int = 100) -> None:
        """
        Visualizes the residual on a 2D slice at constant z.

        Args:
            z_value (float, optional): z-coordinate for the slice. Defaults to 0.0.
            num_points (int, optional): Resolution along x and y axes. Defaults to 100.
        """
        if not self.accelerator.is_local_main_process:
            return

        _x = np.linspace(-1, 1, num_points)
        _y = np.linspace(-1, 1, num_points)
        x, y = np.meshgrid(_x, _y)

        coords = np.stack(
            [x.ravel(), y.ravel(), np.full_like(x.ravel(), z_value)], axis=-1)
        coords_tensor = torch.tensor(
            coords, dtype=torch.float32, device=self.accelerator.device)

        self.model.to(self.accelerator.device)

        r_vals = self.compute_pde_residual(
            coords_tensor).detach().cpu().numpy()

        residual = r_vals.reshape(num_points, num_points)

        plt.figure(figsize=(6, 5))
        plt.pcolormesh(x, y, residual, shading='auto', cmap='coolwarm')
        plt.colorbar(label='Residual')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Residual at z={z_value}')
        plt.tight_layout()
        plt.savefig(self.get_path("residual.png"), dpi=300)
        plt.close()


def exp() -> None:
    """
    Runs hyperparameter experiment: trains models with different
    learning rates, batch sizes, and network architectures.
    """
    num_epochs = 2000

    learning_rates = [0.001, 0.0001, 0.0005, 0.005]
    sample_nums = [1024, 2048, 4096]
    # layer_sizeses = [[3, 512, 512, 512, 1], [
    #     3, 512, 512, 256, 1], [3, 256, 512, 256, 1]]
    layer_sizeses = [[3, 128, 128, 128, 1], [
        3, 256, 256, 256, 1]]

    accelerator = Accelerator()

    for learning_rate, sample_num, layer_sizes in product(learning_rates,
                                                          sample_nums,
                                                          layer_sizeses):
        trainer = PoissonTrainer(layer_sizes, accelerator,
                                 sample_num, learning_rate)

        trainer.train(num_epochs)
        trainer.save_train_plot(
            f"lyrs={layer_sizes}_lr={learning_rate}_bs={sample_num}.png")

        del trainer.model, trainer.optimizer
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()


def run() -> None:
    """
    Training pipeline: trains a single PINN model with default
    hyperparameters and visualizes the results.
    """
    accelerator = Accelerator()

    layer_sizes = [3, 512, 512, 512, 1]
    learning_rate = 0.001
    num_epochs = 2000
    sample_num = 4096

    trainer = PoissonTrainer(layer_sizes, accelerator,
                             sample_num, learning_rate,
                             debug=True)

    trainer.train(num_epochs)

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    """
    Main entrypoint for Acclerator
    """
    run()
    # exp()


if __name__ == "__main__":
    main()
