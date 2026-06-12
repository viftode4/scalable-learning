import os
import torch
import torch.nn as nn
from collections import OrderedDict


def _normalise_sls_lora_init(value):
    if value is None:
        return 'default'
    value = str(value).strip().lower().replace('-', '_')
    if value in ('', 'default', 'none', 'vanilla'):
        return 'default'
    if value in ('orthogonal', 'orthogonal_a', 'orthogonal_lora_a'):
        return 'orthogonal_a'
    if value in ('svd', 'svd_compensated', 'pissa', 'pissa_compensated'):
        return 'svd_compensated'
    raise ValueError(f"unknown SLS_LORA_INIT / sls_lora_init: {value!r}")


def _iter_lora_linear_layers(model):
    for module in model.modules():
        if (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')
                and (hasattr(module, 'base_layer')
                     or callable(getattr(module, 'get_base_layer', None))
                     or hasattr(module, 'weight'))):
            yield module


def _get_lora_base_layer(module):
    get_base_layer = getattr(module, 'get_base_layer', None)
    if callable(get_base_layer):
        return get_base_layer()
    if hasattr(module, 'base_layer'):
        return module.base_layer
    return module


def _get_lora_adapters(module):
    adapters = getattr(module, 'active_adapters', None)
    if adapters:
        return list(adapters)
    adapter = getattr(module, 'active_adapter', None)
    if adapter:
        return [adapter]
    return list(module.lora_A.keys())


def _get_lora_scaling(module, adapter):
    scaling = getattr(module, 'scaling', 1.0)
    if isinstance(scaling, dict):
        scaling = scaling[adapter]
    return float(scaling)


def _get_lora_rank(module, adapter):
    rank = getattr(module, 'r', None)
    if isinstance(rank, dict):
        return int(rank[adapter])
    if rank is not None:
        return int(rank)
    return int(module.lora_A[adapter].weight.shape[0])


def _get_delta_weight(module, adapter):
    get_delta_weight = getattr(module, 'get_delta_weight', None)
    if callable(get_delta_weight):
        return get_delta_weight(adapter)
    return module.lora_B[adapter].weight @ module.lora_A[adapter].weight * \
        _get_lora_scaling(module, adapter)


def _apply_svd_compensated_lora_init(model):
    """Initialise PEFT LoRA factors from top-SVD and compensate base weights.

    For each supported LoRA linear layer we set A/B so PEFT's *scaled* LoRA
    delta equals the rank-r principal reconstruction of the original frozen
    weight W, then subtract that same delta from W. The effective model at
    step 0 is therefore unchanged: (W - delta) + delta == W.
    """
    if getattr(model, '_sls_svd_compensated_lora_init_applied', False):
        raise RuntimeError(
            'SLS_LORA_INIT=svd_compensated already applied to this model')

    count = 0
    max_reconstruction_error = 0.0
    max_delta_norm = 0.0

    with torch.no_grad():
        for module in _iter_lora_linear_layers(model):
            if getattr(module, 'fan_in_fan_out', False):
                raise ValueError(
                    'SLS_LORA_INIT=svd_compensated does not support '
                    'fan_in_fan_out=True LoRA layers yet')

            base_layer = _get_lora_base_layer(module)
            if not hasattr(base_layer, 'weight') or base_layer.weight.ndim != 2:
                raise ValueError(
                    'SLS_LORA_INIT=svd_compensated expects a 2-D base weight')

            for adapter in _get_lora_adapters(module):
                if adapter not in module.lora_A or adapter not in module.lora_B:
                    continue

                base_weight = base_layer.weight
                original = base_weight.detach().clone()
                original_float = original.float().cpu()
                rank = _get_lora_rank(module, adapter)
                max_rank = min(original_float.shape)
                if rank > max_rank:
                    raise ValueError(
                        'SLS_LORA_INIT=svd_compensated rank '
                        f'{rank} exceeds base weight max rank {max_rank}')

                scale = _get_lora_scaling(module, adapter)
                if scale <= 0.0:
                    raise ValueError(
                        'SLS_LORA_INIT=svd_compensated requires positive '
                        f'LoRA scaling, got {scale}')

                u, s, vh = torch.linalg.svd(
                    original_float, full_matrices=False)
                sqrt_s_over_scale = torch.sqrt(s[:rank] / scale)
                b_init = u[:, :rank] * sqrt_s_over_scale.unsqueeze(0)
                a_init = sqrt_s_over_scale.unsqueeze(1) * vh[:rank, :]

                lora_a = module.lora_A[adapter].weight
                lora_b = module.lora_B[adapter].weight
                if tuple(lora_a.shape) != tuple(a_init.shape) or \
                        tuple(lora_b.shape) != tuple(b_init.shape):
                    raise ValueError(
                        'SLS_LORA_INIT=svd_compensated found incompatible '
                        f'LoRA shapes A={tuple(lora_a.shape)}, '
                        f'B={tuple(lora_b.shape)} for base '
                        f'{tuple(base_weight.shape)} and rank {rank}')

                lora_a.copy_(a_init.to(device=lora_a.device,
                                       dtype=lora_a.dtype))
                lora_b.copy_(b_init.to(device=lora_b.device,
                                       dtype=lora_b.dtype))

                delta = _get_delta_weight(module, adapter).detach().to(
                    device=base_weight.device, dtype=base_weight.dtype)
                base_weight.sub_(delta)
                delta_after = _get_delta_weight(module, adapter).detach().to(
                    device=base_weight.device, dtype=base_weight.dtype)
                effective = base_weight.detach().float() + \
                    delta_after.float()
                reconstruction_error = float(
                    (effective - original.float()).abs().max().item())
                max_reconstruction_error = max(max_reconstruction_error,
                                               reconstruction_error)
                max_delta_norm = max(
                    max_delta_norm,
                    float(delta_after.float().norm().item()))
                count += 1

    if count == 0:
        raise RuntimeError(
            'SLS_LORA_INIT=svd_compensated found no PEFT LoRA linear layers')

    model._sls_svd_compensated_lora_init_applied = True
    print("[sls-rolora] SLS_LORA_INIT=svd_compensated: "
          f"initialised {count} LoRA layers; "
          f"max reconstruction error {max_reconstruction_error:.3e}; "
          f"max delta norm {max_delta_norm:.3e}.")
    return model


def _apply_sls_lora_init(model, init_variant):
    """Apply project-specific LoRA init after PEFT creates adapter weights.

    PEFT 0.10.0 does not support PiSSA config switches. Project variants
    therefore run after PEFT has created LoRA layers.
    """
    init_variant = _normalise_sls_lora_init(init_variant)
    if init_variant == 'default':
        return model
    if init_variant == 'svd_compensated':
        return _apply_svd_compensated_lora_init(model)

    if init_variant != 'orthogonal_a':
        raise ValueError(f"unsupported SLS LoRA init: {init_variant!r}")

    count_a = 0
    count_b = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora_A' in name and name.endswith('weight'):
                torch.nn.init.orthogonal_(param)
                count_a += 1
            elif 'lora_B' in name and name.endswith('weight'):
                torch.nn.init.zeros_(param)
                count_b += 1

    print("[sls-rolora] SLS_LORA_INIT=orthogonal_a: "
          f"orthogonalised {count_a} LoRA-A matrices; "
          f"zeroed {count_b} LoRA-B matrices.")
    return model


def enable_adapter(model, package, adapter, **kwargs):
    """
    Enables an adapter for a given model and package.

    Args:
        model: A pre-trained model from HuggingFace Transformers library.
        package: A string indicating the name of the package that provides
            the adapter. Currently, only 'peft' and 'adapterhub' is supported.
        adapter: A string indicating the name of the adapter to enable. The
            available adapters depend on the package.
        **kwargs: Additional keyword arguments that are passed to the
            adapter configuration.

    Returns:
        A model object that has the adapter enabled.

    Raises:
        NotImplementedError: If the package or the adapter is not supported.
    """
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """
        from peft import get_peft_model, TaskType
        if adapter == 'lora':
            sls_lora_init = kwargs.pop(
                'sls_lora_init',
                os.environ.get('SLS_LORA_INIT', 'default'))
            # for name, param in model.named_parameters():
            #     print(name)
            from peft import LoraConfig
            target_modu = [
                "layer.18.attention.self.value","layer.18.attention.self.query",
                "layer.19.attention.self.value","layer.19.attention.self.query",
                "layer.20.attention.self.value","layer.20.attention.self.query",
                "layer.21.attention.self.value","layer.21.attention.self.query",
                "layer.22.attention.self.value","layer.22.attention.self.query",
                "layer.23.attention.self.value","layer.23.attention.self.query",
            ]
            # print("######################################################################")
            # print("######################################################################")
            # print("######################################################################")
            # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["layer.19.attention.self.value","layer.19.attention.self.query","layer.20.attention.self.value","layer.20.attention.self.query","layer.21.attention.self.value","layer.21.attention.self.query","layer.22.attention.self.value","layer.22.attention.self.query","layer.23.attention.self.value","layer.23.attention.self.query"],  **kwargs)
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, **kwargs) # target_modules=target_modu,  **kwargs)

            # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["value"], **kwargs)

            model = get_peft_model(model, peft_config)
            model = _apply_sls_lora_init(model, sls_lora_init)
            # for name, param in model.named_parameters():
            #     print(name)
        elif adapter == 'prefix':
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        model.print_trainable_parameters()

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        # TODO:  After supporting adapterhub, we will move the following
        #   parameters in yaml file for users' convenient
        if adapter == 'lora':
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            from transformers.adapters import AdapterConfig, ConfigUnion

            # TODO: configure these args in cfg
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
    else:
        raise NotImplementedError
    return model


class AdapterModel(nn.Module):
    """
    A wrapper class for a model that can use adapters for fine-tuning.

    This class inherits from torch.nn.Module and implements a wrapper for a
    model that can optionally use adapters for fine-tuning. Adapters are small
    modules that can be inserted between the layers of a pretrained model and
    trained on a specific task, while keeping the original parameters frozen.
    This class can use different adapter packages and methods, such as PEFT
    and LoRA. It also provides methods for saving and loading the model state
    dict, as well as generating text using the model.

    Attributes:
        model: A torch.nn.Module object that represents the original or
            adapted model.

    """
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        """
        Initializes the wrapper with the given model and arguments.

        Args:
            model: A torch.nn.Module object that represents the original model.
            use_adapter: A boolean indicating whether to use adapters for
                fine-tuning. Default is False.
            *args: Additional positional arguments to pass to the adapter
                package or method.
            **kwargs: Additional keyword arguments to pass to the adapter
                package or method. These may include adapter_package,
                adapter_method, etc.
        """
        super().__init__()

        self.model = None
        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            self.model = enable_adapter(model, adapter_package, adapter_method,
                                        **kwargs)
        else:
            self.model = model

    def forward(self, *args, **kwargs):
        """
        Calls the forward method of the wrapped model.

        Args:
            *args: Positional arguments to pass to the model's forward method.
            **kwargs: Keyword arguments to pass to the model's forward method.

        Returns:
            The output of the model's forward method.
        """
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Calls the generate method of the wrapped model.

        Args:
            *args: Positional arguments to pass to the model's generate method.
            **kwargs: Keyword arguments to pass to the model's generate method.

        Returns:
            The output of the model's generate method.
        """
        try:
            res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # When does evaluation in HELM,
            # half precision will cause RuntimeError,
            # the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        return res

    def state_dict(self, return_trainable=True, *args, **kwargs):
        """
        Returns the state dict of the wrapped model.

        Args:
            return_trainable: A boolean indicating whether to return only the
                trainable parameters of the model. Default is True.
            *args: Additional positional arguments to pass to the model's
                state_dict method.
            **kwargs: Additional keyword arguments to pass to the model's
                state_dict method.

        Returns:
            A dictionary containing the state dict of the model. If
            return_trainable is True, only the parameters that require grad are
            included. Otherwise, all parameters are included.
        """
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        """
        Loads the state dict into the wrapped model.

        Args:
            state_dict: A dictionary containing the state dict to load into
                the model.
            strict: A boolean indicating whether to strictly enforce that the
                keys in state_dict match the keys returned by this module’s
                state_dict() function. Default is False.
        """
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        """
        Returns only the trainable parameters of the wrapped model.

        This method can be used to get only the parameters that require grad,
        such as adapters or task-specific layers.

        Returns:
            A dictionary containing the state dict of the trainable parameters
            of the model.
        """
        grad_params = []
        for name, param in self.model.named_parameters():
            #if 'classifier' in name:
            #    print(name)
            #    print(param)
            if param.requires_grad:
                #print(name)
                grad_params.append(name)
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self, path, state=0):
        """
        Saves the model state dict and the current round to a file.

        Args:
            path: A string representing the file path to save the model to.
            state: An integer representing the current round of training or
                evaluation. Default is 0.

        """
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    # TODO: Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)
