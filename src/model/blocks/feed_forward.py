"""Feedforward neural network for feature mapping."""

from torch import Tensor, nn

SLOPE: float = 0.25


class FeedForwardNetwork(nn.Module):
    """A feedforward neural network for feature mapping.."""

    def __init__(self, input_dim: int, expansion_ratio: int = 4, dropout: float = 0.1):
        """
        Initialize the FeedForwardNetwork.

        Args:
            input_dim (int): The dimension of the input feature.
            expansion_ratio (int): The expansion ratio for the hidden layer dimension. Defaults to 4.
            dropout (float): The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self._dim = input_dim
        self._ratio = expansion_ratio
        self._dropout = dropout
        self._expanded_dim = int(input_dim * expansion_ratio)

        self.mapping = nn.Sequential(
            nn.Linear(input_dim, self._expanded_dim),
            nn.PReLU(init=SLOPE),
            nn.Dropout(p=dropout),
            nn.Linear(self._expanded_dim, input_dim),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:  # noqa: WPS231
        """Initialize the parameters of the neural network layers."""
        for module in self.mapping[:-1].modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("leaky_relu", SLOPE))  # type: ignore
                if hasattr(module, "bias") and module.bias is not None:  # noqa: WPS421
                    nn.init.constant_(module.bias, 0)

        # There is no activation after last linear layer.
        for module in self.mapping[-1:].modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:  # noqa: WPS421
                    nn.init.constant_(module.bias, 0)

    def forward(self, embedding: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            embedding (Tensor): The input feature tensor.

        Returns:
            Tensor: The output tensor after passing through the network.
        """
        return self.mapping(embedding)


class LinearLayer(nn.Module):
    """Linear layer configurable with layer normalization, dropout, and non-linearity."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        normilize: bool = True,
        activate: bool = True,
    ) -> None:
        """Initialize the LinearLayer.

        Args:
            input_dim (int): The dimension of the input feature.
            output_dim (int): The dimension of the output feature.
            dropout (float): The dropout probability. Defaults to 0.1.
            normilize (bool): Whether to apply layer normalization. Defaults to True.
            activate (bool): Whether to apply non-linearity. Defaults to True.
        """
        super().__init__()
        norm = nn.LayerNorm(input_dim) if normilize else nn.Identity()
        activation = nn.ReLU() if activate else nn.Identity()
        self.mapper = nn.Sequential(
            *[  # noqa: WPS517
                norm,
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim),
                activation,
            ],
        )

    def forward(self, emb: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            emb (Tensor): The input feature tensor.

        Returns:
            Tensor: The output tensor after passing through the network.
        """
        return self.mapper(emb)  # (N, L, D)


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        normilize: bool = True,
        activate: bool = True,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim (int): The dimension of the input feature.
            hidden_dim (int): The dimension of the hidden layer.
            output_dim (int): The dimension of the output feature.
            num_layers (int): The number of layers in the MLP.
            dropout (float): The dropout probability.
            normilize (bool): Whether to apply layer normalization.
            activate (bool): Whether to apply non-linearity.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        hidden_dims = [hidden_dim for _ in range(num_layers - 1)]
        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]

        list_of_layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            activate = activate if idx < num_layers - 1 else False
            layer = LinearLayer(in_dim, out_dim, dropout, normilize, activate)
            list_of_layers.append(layer)
        self.linear_mapper = nn.Sequential(*list_of_layers)
        self._init_parameters()

    def _init_parameters(self) -> None:  # noqa: WPS231
        """Initialize the parameters of the neural network layers."""
        for module in self.linear_mapper[:-1].modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))  # type: ignore
                if hasattr(module, "bias") and module.bias is not None:  # noqa: WPS421
                    nn.init.constant_(module.bias, 0)

        for module in self.linear_mapper[-1:].modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:  # noqa: WPS421
                    nn.init.constant_(module.bias, 0)

    def forward(self, emb: Tensor) -> Tensor:
        """Forward pass of the network.

        Args:
            emb (Tensor): The input feature tensor.

        Returns:
            Tensor: The output tensor after passing through the network.
        """
        return self.linear_mapper(emb)


class SlimMLP(nn.Module):
    """Very simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim (int): The dimension of the input feature.
            hidden_dim (int): The dimension of the hidden layer.
            output_dim (int): The dimension of the output feature.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        hidden_dims = [hidden_dim for _ in range(num_layers - 1)]
        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]

        list_of_layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            layer = nn.Linear(in_dim, out_dim)
            non_linearity = nn.ReLU(inplace=True) if idx < num_layers - 1 else nn.Identity()  # noqa: WPS221
            list_of_layers.append(layer)
            list_of_layers.append(non_linearity)  # type: ignore
        self.linear_mapper = nn.Sequential(*list_of_layers)
        self._init_parameters()

    def _init_parameters(self) -> None:  # noqa: WPS231
        """Initialize the parameters of the neural network layers."""
        for module in self.linear_mapper[:-1].modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))  # type: ignore
                if hasattr(module, "bias") and module.bias is not None:  # noqa: WPS421
                    nn.init.constant_(module.bias, 0)

        for module in self.linear_mapper[-1:].modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:  # noqa: WPS421
                    nn.init.constant_(module.bias, 0)

    def forward(self, emb: Tensor) -> Tensor:
        """Forward pass of the network.

        Args:
            emb (Tensor): The input feature tensor.

        Returns:
            Tensor: The output tensor after passing through the network.
        """
        return self.linear_mapper(emb)
