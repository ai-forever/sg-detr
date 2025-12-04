"""Transformer decoder based on DAB DETR implementation."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.blocks.feed_forward import SlimMLP
from src.model.blocks.layers import TransformerDecoderLayer
from src.model.blocks.position_encoding import gen_sineembed_for_position
from src.model.utils.model_utils import inverse_sigmoid
from src.model.utils.stacker import get_clones


# pylint: disable=too-many-arguments, too-many-locals
class TransformerDecoder(nn.Module):  # noqa: WPS230
    """Transformer decoder consisting of *args.decoder_layers* layers."""

    def __init__(  # noqa: C901, WPS211
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
        d_model: int = 256,
        query_dim: int = 2,
        temperature: int = 10000,
        predict_quality_score: bool = True,
        init_spans_with_zeros: bool = True,
    ) -> None:
        """
        Initialize a Transformer decoder.

        Args:
            decoder_layer (TransformerDecoderLayer): an instance of the TransformerDecoderLayer() class
            num_layers (int): number of decoder layers
            return_intermediate (bool): whether to return intermediate results
            d_model (int): dimension of the model
            query_dim (int): dimension of the query vector
            temperature (int): temperature of the pos emb.
            predict_quality_score (bool): predict iou of the predicted interval
            init_spans_with_zeros (bool): whether to init last mlp layer with zeros or not
        """
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.temperature = temperature
        self.predict_quality_score = predict_quality_score
        self.init_spans_with_zeros = init_spans_with_zeros
        self.d_model = d_model

        # mlp for query scale
        self.query_scale = SlimMLP(input_dim=d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)

        if predict_quality_score:
            # input_dim = decoder output + anchor embedding
            self.quality_score_embed = SlimMLP(input_dim=d_model * 2, hidden_dim=d_model, output_dim=1, num_layers=3)

        # mlp for query PE embeddings
        self.ref_point_head = SlimMLP(input_dim=d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)

        # modulation mlp
        self.ref_anchor_head = SlimMLP(input_dim=d_model, hidden_dim=d_model, output_dim=1, num_layers=2)

        # anchor regression
        self.span_embed = SlimMLP(input_dim=d_model, hidden_dim=d_model, output_dim=2, num_layers=3)
        self._init_parameters()

        # output norm
        self.norm = nn.LayerNorm(d_model)

    def _init_parameters(self) -> None:
        """Init parameters."""
        # init last reg layer with zeros
        for module in self.span_embed.linear_mapper.modules():
            if isinstance(module, nn.Linear) and module.out_features == 2 and self.init_spans_with_zeros:
                nn.init.constant_(module.weight.data, 0)  # noqa: WPS219
                nn.init.constant_(module.bias.data, 0)  # noqa: WPS219

    def apply_cond_spatial_query(self, query_sine_embed: Tensor, output: Tensor) -> Tensor:
        """Rescale the postional embs leverage conditional spatial query.

        Based on DAB-Implementation.

        Args:
            query_sine_embed (Tensor): PE embedding generated from query embs. Shape: [#Queries, batch_size, dim]
            output (Tensor): Content vector used as query. Shape: [#Queries, batch_size, dim]

        Returns:
            Tensor: Rescaled PE Embs.
        """
        pos_transformation = self.query_scale(output)
        # apply transformation
        return query_sine_embed * pos_transformation

    def update_reference_points(self, output: Tensor, reference_points: Tensor) -> Tensor:
        """Update reference points based on DAB implementation.

        Args:
            output (Tensor): content vector from CA block
            reference_points (Tensor): anchor points

        Returns:
            Tensor: updated reference points
        """
        box_offsets = self.span_embed(output)
        new_boxes = box_offsets[..., : self.query_dim] + inverse_sigmoid(reference_points)  # noqa: WPS221
        return new_boxes.sigmoid()

    def forward(  # noqa: WPS210, C901
        self,
        src: Tensor,
        src_key_padding_mask: Tensor,
        src_pos: Tensor,
        content: Tensor,
        refpoints_unsigmoid: Tensor,
        content_key_padding_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass of the Transformer decoder.

        Args:
            src (Tensor): source video features. Shape: [L_video, batch_size, dim]
            src_key_padding_mask (Tensor): mask for source video features. Shape: [batch_size, L_video]
            src_pos (Tensor): position embeddings for source video features. Shape: [L_video, batch_size, dim]
            content (Tensor): Zero inited embedding represented decoder input. Shape: [#Queries, batch_size, dim]
            refpoints_unsigmoid (Tensor): reference points or anchor points. Shape: [#Queries, batch_size, 2]
            content_key_padding_mask (Optional[Tensor]): mask for content
            content_mask (Optional[Tensor]): mask for content
            src_mask (Optional[Tensor]): mask for source video features

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Decoder outputs from each levele and spans, labels
            and predicted iou score(optional).
        """
        output = content

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        # when predicting the span, the previous anchor is used.
        # Otherwise it will be that the offsets will be predicted twice(one time for anchor, one time for span), For
        # this reason, anchors from 0 to the number of decoder layers -1 are returned
        ref_points = [reference_points]
        quality_scores: List[Tensor] = []

        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(reference_points, self.d_model, temperature=self.temperature)

            # construct PE emb for self attention layer
            query_pos = self.ref_point_head(query_sine_embed)

            query_sine_embed = self.apply_cond_spatial_query(query_sine_embed, output)

            # modulated HW attentions
            reft_cond = self.ref_anchor_head(output).sigmoid().squeeze(2)  # nq, bs, 1
            obj_width = reference_points[..., 1]
            modulation_value = (reft_cond / obj_width).unsqueeze(-1)  # noqa: WPS221
            query_sine_embed = query_sine_embed * modulation_value  # noqa: WPS350

            output = layer(
                tgt=output,
                src=src,
                query_sine_embed=query_sine_embed,
                query_pos=query_pos,
                src_pos=src_pos,
                tgt_mask=content_mask,
                src_mask=src_mask,
                tgt_key_padding_mask=content_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            # update anchor
            new_reference_points = self.update_reference_points(output, reference_points)

            if self.predict_quality_score:
                #  iou score is predicted by prediction and therefore the gradient should not flow by prediction
                reference_points_embed = gen_sineembed_for_position(
                    new_reference_points,
                    self.d_model,
                    temperature=self.temperature,
                ).detach()
                # the decoder outputs and the predicted span are used for prediction iou score
                score_data = torch.concat([output, reference_points_embed], dim=-1)
                quality_score = self.quality_score_embed(score_data)

            # Note: anchors are always used with detach (except for the first trainable anchor), since they must be
            # constants when predicting offset
            reference_points = new_reference_points.detach()

            # do not return anchors from the last layer
            if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.predict_quality_score:
                    quality_scores.append(quality_score)

        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)
            stacked_decoder_outputs = torch.stack(intermediate).transpose(1, 2)
            stacked_reference_points = torch.stack(ref_points).transpose(1, 2)
            if self.predict_quality_score:
                stacked_quality_scores = torch.stack(quality_scores).transpose(1, 2)
            else:
                stacked_quality_scores = None
            return stacked_decoder_outputs, stacked_reference_points, stacked_quality_scores
        if self.predict_quality_score:
            return output.unsqueeze(0), new_reference_points.unsqueeze(0), quality_score.unsqueeze(0)
        return output.unsqueeze(0), new_reference_points.unsqueeze(0), None
