import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import wandb
from src.utils.scaled_grid_tools import get_grid_block_offsets, average_grid_cells

# Currently unnneded
# from collections import OrderedDict
# from typing import Any
# import pandas as pd
# from sklearn.metrics import roc_auc_score
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
# from torchmetrics.image import PeakSignalNoiseRatio as PSNR

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.model.module_discriminator import My_Discriminator_v1
from src.model.module_my_stacked_unet import StackedUNet
from src.model.module_miniconv import MiniConvNet

def coord_channels(B, W, H):
    xx_channel = torch.linspace(-1, 1, W).repeat(H, 1)
    yy_channel = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W)
    
    xx_channel = xx_channel.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1)
    yy_channel = yy_channel.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1)

    x = torch.cat([xx_channel, yy_channel], dim=1)
    return x


def batch_mse(prediction, target):
    return ((prediction - target) ** 2).mean(dim=(1, 2, 3))

def get_predefined_network(name, in_channels, out_channels, network_kwargs):
    if name == "unet_resnet50":
        resnet_encoder_depth = 5  # Default encoder depth for resnet50
        filters = [2 ** (10 - i) for i in range(resnet_encoder_depth)]  # Dynamically set filters
        network = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
            encoder_depth=resnet_encoder_depth,
            decoder_channels=filters,
        )
        # Add dropout to all decoder blocks
        for block in network.decoder.blocks:
            block.conv1.add_module("dropout", nn.Dropout2d(p=network_kwargs["dropout_p"]))
    elif name == "unet_mobilenetv2":
        resnet_encoder_depth = 5
        filters = [2 ** (10 - i) for i in range(resnet_encoder_depth)]  # Dynamically set filters
        network = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
            encoder_depth=resnet_encoder_depth,
            decoder_channels=filters,
        )
    elif name == "my_stacked_unet":
        network = StackedUNet(
            in_channels=in_channels,
            out_channels=out_channels)
    elif name == "miniconv":
        # Use MiniConvNet as encoder
        network = MiniConvNet(
            in_channels=in_channels,
            num_filters=16,
            out_channels=out_channels,
            kernel_size=3,
            num_layers=4
        )
    return network

class SpatialAverageLayer(nn.Module):
    def forward(self, x):
        if x.dim() == 4:
            avg = x.mean(dim=[2, 3], keepdim=True)
        elif x.dim() == 3:
            avg = x.mean(dim=[1, 2], keepdim=True)
        else:
            raise ValueError("Input must be 3D or 4D tensor (C, H, W) or (B, C, H, W)")
        return avg.expand_as(x)

class ImageTRANSIT(LightningModule):
    def __init__(self, 
                 # Different channels for different inputs
                 image_chan=1,
                 latent_chan=1, 
                 generator_noise_chan=0,  # Number of noise channels for the generator
                 condition_chan=3, 
                 
                 conditional=False, # Swith to make generator conditional by adding "c1emb" to the input
                 conditional_encoder=True, # Swith to make generator conditional by adding "c2emb" to the input
                 
                 learning_rate=1e-3, 
                 lr_scheduler=None,
                 dropout_p=0.0, 
                 encoder_end_activation="sigmoid",
                 generator_end_activation="sigmoid",
                 
                 adversarial=False, 
                 discriminator_lr=1e-3,
                 maximum_classes=30,  # Maximum number of classes for one-hot encoding
                 w_l2=1,
                 w_adv=None, # Default to None, will intentionally produce an error if used but not specified in the constructor
                 samples_to_plot=5,
                 condition_in_discriminator=True,
                 
                 generator_net="unet_resnet50",
                 encoder_net=None,
                 
                 encoder_train_direct_MSE=False,
                 do_validate_encoder=True,  # New flag for encoder training
                 
                 spatial_average_encode_output=False, 
                 encoder_concat_in=["x2"],
                 generator_concat_in=["zt"],
                 encoder_append_layers=None,
                 generator_append_layers=None,
                 ):  # Kernel size for MiniConvNet
        """
        ImageTRANSIT model for transforming "zt" into an output image.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            learning_rate (float): Learning rate for the generator.
            dropout_p (float): Dropout probability for the decoder.
            end_activation (callable): Activation function to apply to the output (default: F.relu).
            conditional (bool): Whether the model is conditional on "c1".
            condition_chan (int): Dimension of the condition embedding.
            adversarial (bool): Whether to train adversarially using a discriminator.
            discriminator_lr (float): Learning rate for the discriminator.
        """
        super(ImageTRANSIT, self).__init__()
        
        self.channel_dims = {"x1": image_chan, 
                             "x2": image_chan, 
                             "zt": image_chan, 
                             "enc_out": latent_chan, 
                             "c1emb": condition_chan, 
                             "c2emb": condition_chan, 
                             "x1_hat": image_chan, 
                             "gn": generator_noise_chan, 
                             "coord_ch": 2}
        
        self.save_hyperparameters(logger=False)
        
        # Decide what networks are going to be trained
        self.use_encoder = True if encoder_net is not None else False
        self.use_generator = True if generator_net is not None else False
        self.adversarial = adversarial
        
        self.learning_rate = learning_rate
        self.discriminator_lr = discriminator_lr
        self.criterion = nn.MSELoss()
        # self.end_activation = end_activation  # Remove this
        self.encoder_end_activation = encoder_end_activation
        self.generator_end_activation = generator_end_activation
        
        # Solve the conditional buisiness
        self.conditional = conditional
        self.conditional_encoder = conditional_encoder  # Whether to use condition in the encoder
        if self.conditional and not "c1emb" in generator_concat_in:
            generator_concat_in.append("c1emb")  # Ensure condition is included in encoder input if conditional
            print("WARNING: Condition 'c1emb' automatically added to generator inputs:", encoder_concat_in)
        if self.conditional_encoder and not "c2emb" in encoder_concat_in:
            encoder_concat_in.append("c2emb")
            print("WARNING: Condition 'c2emb' automatically added to encoder inputs:", generator_concat_in)
        self.condition_in_discriminator = condition_in_discriminator # TODO: make this consistent with the rest of the code
        

        self.maximum_classes = maximum_classes  # Maximum number of classes for one-hot encoding
        self.w_l2 = w_l2
        self.w_adv = w_adv
        self.samples_to_plot = samples_to_plot
        self.encoder_train_direct_MSE = encoder_train_direct_MSE
        #self.combine_encoder_generator_into_transport_model = combine_encoder_generator_into_transport_model
        self.generator_noise_chan = generator_noise_chan  # Number of noise channels for the generator
        self.spatial_average_encode_output = spatial_average_encode_output
        self.do_validate_encoder = do_validate_encoder  # Flag to control encoder validation
        self.encoder_concat_in = encoder_concat_in  # List of inputs to concatenate in the encoder
        self.generator_concat_in = generator_concat_in  # List of inputs to concatenate in the generator
        self.encoder_append_layers = encoder_append_layers[:] if encoder_append_layers is not None else []
        self.generator_append_layers = generator_append_layers[:] if generator_append_layers is not None else []

        def _activation_in_layers(append_layers, activation):
            if activation is None:
                return True
            if not isinstance(activation, str):
                return False
            activation = activation.lower()
            for l in append_layers:
                if isinstance(l, str) and l.lower() == activation:
                    return True
            return False

        # Only append activation if not present and is a string
        if isinstance(encoder_end_activation, str) and not _activation_in_layers(self.encoder_append_layers, encoder_end_activation):
            self.encoder_append_layers.append(encoder_end_activation)
        if isinstance(generator_end_activation, str) and not _activation_in_layers(self.generator_append_layers, generator_end_activation):
            self.generator_append_layers.append(generator_end_activation)

        if self.use_generator:
            # Get the number of in and out channels for the generator
            in_channels_generator = self.get_concat_channels_dim(self.generator_concat_in)  # Concatenate channels from specified inputs
            out_channels_generator = self.channel_dims["x1_hat"]  # Output channels should match the input channels
            
            # Get the network
            if isinstance(generator_net, str):
                # If generator_net is a string, get the predefined network
                self.generator = get_predefined_network(generator_net, in_channels_generator, out_channels_generator, {"dropout_p": dropout_p})
            else:
                # Generator is a "partial" defined in hydra configs, mening we only have to provide in_channels and out_channels
                self.generator = generator_net(in_channels=in_channels_generator, out_channels=out_channels_generator)
            
            # Make sure ReLU is not inplace
            for module in self.generator.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
            # Append generator end layers as specified
            if self.generator_append_layers:
                layers = [self.generator] + [self._as_module(l, out_channels_generator) for l in self.generator_append_layers]
                self.generator = nn.Sequential(*layers)
        
        if self.use_encoder:
            in_channels_encoder = self.get_concat_channels_dim(self.encoder_concat_in)  # Concatenate channels from specified inputs
            out_channels_encoder = self.channel_dims["enc_out"]  # Output channels should match the latent channels
            
            # Get the network
            if isinstance(encoder_net, str):
                # If encoder_net is a string, get the predefined network
                self.encoder = get_predefined_network(encoder_net, in_channels_encoder, out_channels_encoder, {"dropout_p": dropout_p})
            else:
                # Generator is a "partial" defined in hydra configs, mening we only have to provide in_channels and out_channels
                self.encoder = generator_net(in_channels=in_channels_encoder, out_channels=out_channels_encoder)

            # Make sure ReLU is not inplace
            for module in self.encoder.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
            # Append encoder end layers as specified
            if self.encoder_append_layers:
                layers = [self.encoder] + [self._as_module(l, out_channels_encoder) for l in self.encoder_append_layers]
                self.encoder = nn.Sequential(*layers)
        # Conditional processing
        if self.conditional:
            self.condition_fc = nn.Linear(self.maximum_classes, condition_chan)

        # Adversarial training
        if self.adversarial:
            if self.condition_in_discriminator and self.conditional:
                # If condition is included in discriminator, adjust input channels accordingly
                self.discriminator = My_Discriminator_v1(self.get_concat_channels_dim(["x1", "zt", "c1emb"]))
            else:
                self.discriminator = My_Discriminator_v1(self.get_concat_channels_dim(["x1", "zt"]))
            self.adversarial_loss = nn.BCELoss()
        self.automatic_optimization = not adversarial  # Enable manual optimization if adversarial training is used

    def get_concat_channels_dim(self, instance_list):
        channel_dim = 0
        for instance in instance_list:
            if instance in self.channel_dims:
                channel_dim += self.channel_dims[instance]
            else:
                raise ValueError(f"Instance '{instance}' not found in channel dimensions: {self.channel_dims}")
        return channel_dim
    
    def embed_condition(self, c1, im_width=128, im_height=128):
        """
        Embed the condition tensor into a suitable format for the model.

        Args:
            c1 (torch.Tensor): Condition tensor (one-hot encoded).

        Returns:
            torch.Tensor: Embedded condition tensor.
        """
        if self.conditional and c1 is not None:
            condition = c1.int().long()
            condition = F.one_hot(condition, num_classes=self.maximum_classes)  # One-hot encode the condition
            condition = self.condition_fc(condition.float())  # Pass through a linear layer
            condition = condition.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            condition = condition.expand(-1, -1, im_width, im_height)  # Tile to match image size
            return condition
        return None

    def _as_module(self, layer, num_channels=None):
        """Wrap a function, string, or module as nn.Module. Pass num_channels for batchnorm."""
        if isinstance(layer, nn.Module):
            return layer
        elif callable(layer):
            class LambdaLayer(nn.Module):
                def __init__(self, func):
                    super().__init__()
                    self.func = func
                def forward(self, x):
                    return self.func(x)
            return LambdaLayer(layer)
        elif isinstance(layer, str):
            if layer.lower() == "batchnorm":
                if num_channels is None:
                    raise ValueError("num_channels must be provided for batchnorm.")
                return nn.BatchNorm2d(num_channels)
            elif layer.lower() == "spatial_average":
                return SpatialAverageLayer()
            else:
                # Use a mapping for activations
                activations = {
                    "sigmoid": nn.Sigmoid,
                    "relu": nn.ReLU,
                    "leakyrelu": nn.LeakyReLU,
                    "tanh": nn.Tanh,
                    "softmax": nn.Softmax,
                    "softplus": nn.Softplus,
                    "elu": nn.ELU,
                    "selu": nn.SELU,
                    "gelu": nn.GELU,
                    # Add more if needed
                }
                key = layer.lower()
                if key in activations:
                    return activations[key]()
                else:
                    raise ValueError(f"Unknown layer string: {layer}")
        else:
            raise ValueError("Layer must be callable, nn.Module, or a supported string.")

    def action_generate(self, zt, c1=None, enc_out=None, noise=None):
        """
        Forward pass of the generator in ImageTRANSIT model.

        Args:
            zt (torch.Tensor): Input tensor (template block).
            c1 (torch.Tensor, optional): Condition tensor (one-hot encoded).

        Returns:
            torch.Tensor: Output tensor.
        """
        assert self.use_generator, "Generator is not defined. Please specify a generator during initialization."
        
        gen_input_list = []
        for instance in self.generator_concat_in:
            if instance == "zt":
                gen_input_list.append(zt)
            elif instance == "c1emb" and c1 is not None:
                condition = self.embed_condition(c1, im_width=zt.size(2), im_height=zt.size(3))
                gen_input_list.append(condition)
            elif instance == "gn":
                if noise is None:
                    noise = torch.normal(0, 1, size=(zt.size(0), self.generator_noise_chan, zt.size(2), zt.size(3)), device=zt.device)
                gen_input_list.append(noise)
            elif instance == "enc_out":
                gen_input_list.append(enc_out)
            elif instance == "coord_ch":
                # Add coordinate channels
                coord_channels_tensor = coord_channels(zt.size(0), zt.size(2), zt.size(3))
                gen_input_list.append(coord_channels_tensor.to(zt.device))
            else:
                raise ValueError(f"Instance '{instance}' not recognized in valid generator inputs")
        
        generator_input = torch.cat(gen_input_list, dim=1)  # Concatenate inputs along the channel dimension
        
        x = self.generator(generator_input)
        return x

    def action_encode(self, x2, c2=None, zt=None, quantize_encoder_output=False):
        """
        Encode the input image using the encoder if defined.

        Args:
            x2 (torch.Tensor): Input tensor (image to encode).
            c2 (torch.Tensor, optional): Condition tensor (one-hot encoded).

        Returns:
            torch.Tensor: Encoded representation of the input image.
        """
        assert self.use_encoder, "Encoder is not defined. Please specify an encoder during initialization."
        
        enc_input_list = []
        for instance in self.encoder_concat_in:
            if instance == "zt" and zt is not None:
                enc_input_list.append(zt)
            elif instance == "x2":
                enc_input_list.append(x2)
            elif instance == "c2emb" and c2 is not None:
                condition = self.embed_condition(c2, im_width=x2.size(2), im_height=x2.size(3))
                enc_input_list.append(condition)
            elif instance == "coord_ch":
                # Add coordinate channels
                coord_channels_tensor = coord_channels(x2.size(0), x2.size(2), x2.size(3))
                enc_input_list.append(coord_channels_tensor.to(x2.device))
            else:
                raise ValueError(f"Instance '{instance}' not recognized in valid encoder inputs")
        
        encoder_input = torch.cat(enc_input_list, dim=1)  # Concatenate inputs along the channel dimension

        encoded_output = self.encoder(encoder_input)
        if quantize_encoder_output:
            encoded_output = (encoded_output > 0.5).float()

        return encoded_output
    
    def forward(self, def_inp, c1=None):
        """
        Forward pass of the ImageTRANSIT model.

        Args:
            def_inp (torch.Tensor): default Input tensor for the model configuration.
            c1 (torch.Tensor, optional): Condition tensor (one-hot encoded).

        Returns:
            torch.Tensor: Output tensor.
        """
        # be default, use the generate method
        if self.use_generator:
            return self.action_generate(def_inp, c1)
        else:
            # If generator is not defined, use the encoder to encode zt
            if self.use_encoder:
                return self.action_encode(def_inp, c1, quantize_encoder_output=False)
            else:
                raise ValueError("Generator and Encoder are not defined. Please specify at least one during initialization.")
    
    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (dict): Batch of data containing "zt", "x1", and optionally "c1".
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        zt = batch["zt"]  # Template block
        x1 = batch["x1"]  # Captured block
        c1 = batch.get("c1", None)  # Condition (optional)

        # Solo encoder training
        # Train the encoder to reconstruct zt from x1
        if self.encoder_train_direct_MSE:
            enc_out = self.action_encode(x1, c1)
            encoder_loss = torch.mean((enc_out-zt)**2)
            self.log("encoder_train_loss", encoder_loss, batch_size=x1.size(0))
        else:
            encoder_loss = 0


        # generator training (with or without encoder)
        if self.use_generator:
            if self.use_encoder:
                latent = self.action_encode(batch["x2"], batch.get("c2", None), zt, quantize_encoder_output=False)
            else:
                latent = None
            
            # Forward pass of generator
            generated = self.action_generate(zt, c1, enc_out=latent)

            if self.adversarial: # DOES NOT WORK, NEEDS FIXING
                # Access optimizers manually
                generator_optimizer, discriminator_optimizer = self.optimizers()

                # Train discriminator
                discriminator_optimizer.zero_grad()
                valid = torch.ones((x1.size(0), 1), device=self.device)  # Real labels
                fake = torch.zeros((x1.size(0), 1), device=self.device)  # Fake labels

                # Real loss
                real_inputs_list = [zt, x1]
                if self.conditional and self.condition_in_discriminator:
                    real_inputs_list.append(self.embed_condition(c1, im_width=zt.size(2), im_height=zt.size(3)))
                pred_real = self.discriminator(torch.cat(real_inputs_list, dim=1))

                real_loss = self.adversarial_loss(pred_real, valid)

                # Fake loss
                fake_inputs_list = [zt, generated.detach()]  # Detach to avoid backpropagation through discriminator
                if self.conditional and self.condition_in_discriminator:
                    fake_inputs_list.append(self.embed_condition(c1, im_width=zt.size(2), im_height=zt.size(3)))
                pred_fake = self.discriminator(torch.cat(fake_inputs_list, dim=1))
                fake_loss = self.adversarial_loss(pred_fake, fake)

                # Combined discriminator loss
                d_loss = (real_loss + fake_loss) / 2
                self.manual_backward(d_loss)
                discriminator_optimizer.step()
                self.log("discriminator_loss", d_loss)

                # Train generator
                generator_optimizer.zero_grad()
                fake_inputs_list_gen = [zt, generated]  # Do not detach to allow backpropagation
                if self.conditional and self.condition_in_discriminator:
                    fake_inputs_list_gen.append(self.embed_condition(c1, im_width=zt.size(2), im_height=zt.size(3)))
                pred_fake = self.discriminator(torch.cat(fake_inputs_list_gen, dim=1))
                g_loss = self.adversarial_loss(pred_fake, valid)

                # Reconstruction loss
                mse_loss = self.criterion(generated, x1)

                # Combined generator loss
                generator_loss = mse_loss * self.w_l2 + g_loss * self.w_adv
                self.manual_backward(generator_loss + encoder_loss)  # Combine encoder and generator losses
                generator_optimizer.step()
                self.log("train_loss", generator_loss + encoder_loss)

                return generator_loss + encoder_loss

            else:
                # Standard training without adversarial loss
                mse_loss = self.criterion(generated, x1)
                loss = mse_loss + encoder_loss  # Combine encoder and generator losses
                self.log("train_loss", loss, batch_size=x1.size(0))
                return loss
        else:
            loss = encoder_loss  # Combine encoder and generator losses
            self.log("train_loss", loss, batch_size=x1.size(0))
            return encoder_loss

    def action_transport(self, x2, c2=None, c1=None, zt=None):
        latent = self.action_encode(x2=x2, c2=c2, zt=zt, quantize_encoder_output=False) if self.use_encoder else None
        output = self.action_generate(zt, c1=c1, enc_out=latent)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (dict): Batch of data containing "zt", "x1", and optionally "c1".
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        
        zt = batch["zt"]
        x1 = batch["x1"]
        ids = batch["ids"]  # Image IDs
        parts = batch["parts"]  # Block IDs
        c1 = batch.get("c1", None)  # Condition (optional)

        # Forward pass
        if self.use_generator:
            if self.use_encoder:
                enc_out = self.action_encode(batch["x2"], batch.get("c2", None), zt, quantize_encoder_output=False)
            else:
                enc_out = None
            
            # Forward pass of generator
            generated = self.action_generate(zt, c1, enc_out=enc_out)
        else:
            if self.use_encoder:
                enc_out = self.action_encode(x1, c1, quantize_encoder_output=False)
        

        # Plot and log to WnB during a random batch
        if batch_idx == 0:  # Random batch visualization
            self.validation_plot(batch=batch, 
                            zt=zt, 
                            x1=x1, 
                            generated=generated if self.use_generator else None, 
                            enc_out=enc_out if self.use_encoder else None, 
                            c1=c1, 
                            ids=ids, 
                            parts=parts)

        # Compute the zt reconstruction quality
        if self.use_encoder and self.do_validate_encoder:
            error_rate = 0.0
            for i in range(zt.size(0)):
                block_id = int(parts[i][6:])  # Assuming parts[i] corresponds to the block_id
                x_offset, y_offset = get_grid_block_offsets(
                    block_id=block_id,
                    x_scale=3,
                    y_scale=3,
                    block_stride_w=64,
                    block_stride_h=64,
                    block_w=128,
                    block_h=128,
                    post_scale_w=684,
                    post_scale_h=684
                )
                
                # Average grid cells in enc_out
                enc_out_np = enc_out[i, 0].detach().cpu().numpy()
                averaged_enc_out = average_grid_cells(enc_out_np, scale=3, x_offset=x_offset, y_offset=y_offset)
                quantized_reconstruction = (averaged_enc_out > 0.5).astype(np.float32)

                # Compute difference and error map
                zt_np = zt[i, 0].cpu().numpy()
                zt_reco_difference = zt_np - quantized_reconstruction
                error_rate += np.mean(np.abs(zt_reco_difference))
            error_rate /= zt.size(0)
            
            mse_zt_reco = ((zt - enc_out) ** 2).mean().item()
            self.log("zt_reco_error_rate", error_rate)
            self.log("zt_reco_mse", mse_zt_reco)

        main_opt_loss=0
        # Get the generator related losses
        if self.use_generator:
            # Compute CDP generation MSE loss
            mse_loss = self.criterion(generated, x1)
            self.log("val_mse_loss", mse_loss)

            if self.adversarial:
                # Compute adversarial loss
                valid = torch.ones((x1.size(0), 1), device=self.device)  # Real labels

                # Prepare inputs for discriminator
                pred_fake_inputs_list = [zt, generated]
                if self.conditional and self.condition_in_discriminator:
                    pred_fake_inputs_list.append(self.embed_condition(c1, im_width=zt.size(2), im_height=zt.size(3)))
                pred_fake = self.discriminator(torch.cat(pred_fake_inputs_list, dim=1))

                adv_loss = self.adversarial_loss(pred_fake, valid)
                self.log("val_adv_loss", adv_loss)

                # Combined validation loss
                generator_loss = mse_loss * self.w_l2 + adv_loss * self.w_adv
            else:
                # Standard validation loss
                generator_loss = mse_loss
        else:
            generator_loss = 0
        
        #Get the ecoder related loss
        if self.encoder_train_direct_MSE:
            encoder_loss = torch.mean((enc_out-zt)**2)
        else:
            encoder_loss = 0
            
        main_opt_loss=generator_loss+encoder_loss
        self.log("val_loss", main_opt_loss)

        return main_opt_loss


    def validation_plot(self, batch, zt, x1, generated=None, enc_out=None, c1=None, ids=None, parts=None):
        # Define columns to plot
        columns_to_plot = ["zt", "x1"]
        if "x2" in batch:
            columns_to_plot.append("x2")
        if self.use_generator:
            columns_to_plot += ["generated", "x1-generated"]
        if self.use_encoder:
            columns_to_plot += ["zt_reconstruction"]
        # if not self.combine_encoder_generator_into_transport_model:
        #     columns_to_plot+=["zt_reco_q"]
        
        num_samples = min(self.samples_to_plot, zt.size(0))  # Plot up to samples_to_plot samples
        fig, axes = plt.subplots(num_samples, len(columns_to_plot), figsize=(4 * len(columns_to_plot), 4 * num_samples))

        for i in range(num_samples):
            for j, column in enumerate(columns_to_plot):
                if column == "zt":
                    axes[i, j].imshow(zt[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, j].set_title(f"zt (Template)\nID: {ids[i]}, Part: {parts[i]}")
                elif column == "x1":
                    axes[i, j].imshow(x1[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, j].set_title(f"x1 (Captured, c1={c1[i].item() if c1 is not None else 'N/A'})")
                elif column == "x2":
                    axes[i, j].imshow(batch["x2"][i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, j].set_title(f"x2 (Captured, c2={batch.get('c2', [None])[i]})")
                elif column == "generated":
                    axes[i, j].imshow(generated[i, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, j].set_title("Generated")
                elif column == "x1-generated":
                    diff = (x1[i, 0] - generated[i, 0]).detach().cpu().numpy()
                    mse_diff = ((x1[i, 0] - generated[i, 0]) ** 2).mean().item()
                    im = axes[i, j].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
                    axes[i, j].set_title(f"Difference (x1 - Output)\nMSE: {mse_diff:.4f}")
                elif column == "zt_reconstruction" and self.use_encoder:
                    reconstructed_mse = ((zt[i, 0] - enc_out[i, 0]) ** 2).mean().item()
                    axes[i, j].imshow(enc_out[i, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, j].set_title(f"Reconstructed zt\nMSE: {reconstructed_mse:.4f}")
                elif column == "zt_reco_q" and self.use_encoder:
                    # Get block offsets using block_id derived from part
                    block_id = int(parts[i][6:])  # Assuming parts[i] corresponds to the block_id
                    x_offset, y_offset = get_grid_block_offsets(
                        block_id=block_id,
                        x_scale=3,
                        y_scale=3,
                        block_stride_w=64,
                        block_stride_h=64,
                        block_w=128,
                        block_h=128,
                        post_scale_w=684,
                        post_scale_h=684
                    )

                    # Average grid cells in enc_out
                    enc_out_np = enc_out[i, 0].detach().cpu().numpy()
                    averaged_enc_out = average_grid_cells(enc_out_np, scale=3, x_offset=x_offset, y_offset=y_offset)
                    quantized_reconstruction = (averaged_enc_out > 0.5).astype(np.float32)

                    # Compute difference and error map
                    zt_np = zt[i, 0].cpu().numpy()
                    zt_reco_difference = zt_np - quantized_reconstruction
                    error_rate = np.mean(np.abs(zt_reco_difference))
                    
                    error_map = np.zeros_like(zt_reco_difference, dtype=np.float32)
                    error_map[zt_reco_difference == 1] = 0.5  # 1->0 bit error (red)
                    error_map[zt_reco_difference == -1] = 1.0  # 0->1 bit error (blue)
                    

                    # Plot quantized reconstruction and error map
                    axes[i, j].imshow(quantized_reconstruction, cmap="gray", vmin=0, vmax=1)
                    axes[i, j].imshow(error_map, cmap="coolwarm", alpha=np.abs(error_map))  # Overlay error map
                    axes[i, j].set_title(f"x_offset{x_offset}y_offset{y_offset}\nError Rate: {error_rate:.4f}") #Quantized zt Reconstruction
                axes[i, j].axis("off")

        plt.suptitle(f"Validation Epoch {self.current_epoch}")
        plt.tight_layout()

        # Save the plot to WnB
        if wandb.run is not None:
            wandb.log({"Validation Plot": wandb.Image(fig)})
        plt.close(fig)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            list: List of optimizers.
        """
        parameters = []
        if self.use_generator:
            parameters += list(self.generator.parameters())
        if self.use_encoder:
            parameters += list(self.encoder.parameters())
        generator_optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        if self.adversarial:
            discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr)
            if hasattr(self.hparams, 'lr_scheduler'):
                print("Using learning rate scheduler for generator, {:}".format(self.hparams.lr_scheduler))
                print("Using learning rate scheduler for discriminator, {:}".format(self.hparams.lr_scheduler))
                generator_scheduler = self.hparams.lr_scheduler(generator_optimizer)
                discriminator_scheduler = self.hparams.lr_scheduler(discriminator_optimizer)
                return [generator_optimizer, discriminator_optimizer], [generator_scheduler, discriminator_scheduler]
            else:
                return [generator_optimizer, discriminator_optimizer]
        
        if hasattr(self.hparams, 'lr_scheduler') and self.hparams.lr_scheduler is not None:
            print("Using learning rate scheduler for generator, {:}".format(self.hparams.lr_scheduler))
            generator_scheduler = self.hparams.lr_scheduler(generator_optimizer)
            return [generator_optimizer], [generator_scheduler]

        return generator_optimizer

    # def on_validation_epoch_end(self) -> None:
    #     """Makes several plots of the jets and how they are reconstructed.
    #     """
    #     schedulers = self.lr_schedulers()
    #     if schedulers is not None:
    #         # Handle both single scheduler and list of schedulers
    #         if isinstance(schedulers, list):
    #             for sched in schedulers:
    #                 sched.step()
    #         else:
    #             schedulers.step()


if __name__ == "__main__":
    # Example usage
    from pytorch_lightning import Trainer
    from src.data.data import CDPDatamodule

    # Dataset path
    dataset_path = "data/cdp_transit_dataset"

    # Initialize datamodule
    datamodule = CDPDatamodule(dataset_path, batch_size=16, n_captures=1)

    # Initialize model
    model = ImageTRANSIT(in_channels=1, out_channels=1, learning_rate=1e-3)

    # Train the model
    trainer = Trainer(
        max_epochs=10,
        accelerator="cpu",
    )
    trainer.fit(model, datamodule=datamodule)

