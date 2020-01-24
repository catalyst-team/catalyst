from typing import List, Optional, Union, Dict  # isort:skip

from utils.typing import Device, Model

from catalyst.dl.runner import GanRunner


class WGanRunner(GanRunner):
    """
    Wasserstein GAN Runner
        Note: all the changes made compared to GANRunner
            are just renaming some input/output keys to conventional ones
    Also this runner may be used unchanged for WGAN-GP
        (just add gradient penalty loss in yaml config)
    """
    def __init__(
        self,
        model: Union[Model, Dict[str, Model]] = None,
        device: Device = None,
        input_batch_keys: Optional[List[str]] = None,
        data_input_key: str = "data",
        class_input_key: str = "class_targets",
        noise_input_key: str = "noise",
        fake_logits_output_key: str = "fake_validity",
        real_logits_output_key: str = "real_validity",
        fake_data_output_key: str = "fake_data",
        condition_keys: Optional[Union[str, List[str]]] = None,
        d_fake_condition_keys: List[str] = None,
        d_real_condition_keys: List[str] = None,
        generator_train_phase: str = "generator_train",
        discriminator_train_phase: str = "critic_train",
        generator_model_key: str = "generator",
        discriminator_model_key: str = "critic"
    ):
        super().__init__(
            model=model,
            device=device,
            input_batch_keys=input_batch_keys,
            data_input_key=data_input_key,
            class_input_key=class_input_key,
            noise_input_key=noise_input_key,
            fake_logits_output_key=fake_logits_output_key,
            real_logits_output_key=real_logits_output_key,
            fake_data_output_key=fake_data_output_key,
            condition_keys=condition_keys,
            d_fake_condition_keys=d_fake_condition_keys,
            d_real_condition_keys=d_real_condition_keys,
            generator_train_phase=generator_train_phase,
            discriminator_train_phase=discriminator_train_phase,
            generator_model_key=generator_model_key,
            discriminator_model_key=discriminator_model_key
        )


class CGanRunner(GanRunner):
    """
    (Class) Conditional GAN
        both generator and discriminator are conditioned on one-hot class target
    """
    def __init__(
        self,
        model: Union[Model, Dict[str, Model]] = None,
        device: Device = None,
        data_input_key: str = "data",
        class_input_key: str = "class_targets",
        noise_input_key: str = "noise",
        fake_logits_output_key: str = "fake_logits",
        real_logits_output_key: str = "real_logits",
        fake_data_output_key: str = "fake_data",
        d_fake_condition_key: str = "class_targets_one_hot",
        d_real_condition_key: str = "class_targets_one_hot",
        generator_train_phase: str = "generator_train",
        discriminator_train_phase: str = "discriminator_train",
        generator_model_key: str = "generator",
        discriminator_model_key: str = "discriminator"
    ):
        input_batch_keys = [data_input_key, class_input_key]
        condition_keys = [d_fake_condition_key]
        d_fake_condition_keys = [d_fake_condition_key]
        d_real_condition_keys = [d_real_condition_key]
        super().__init__(
            model=model,
            device=device,
            input_batch_keys=input_batch_keys,
            data_input_key=data_input_key,
            class_input_key=class_input_key,
            noise_input_key=noise_input_key,
            fake_logits_output_key=fake_logits_output_key,
            real_logits_output_key=real_logits_output_key,
            fake_data_output_key=fake_data_output_key,
            condition_keys=condition_keys,
            d_fake_condition_keys=d_fake_condition_keys,
            d_real_condition_keys=d_real_condition_keys,
            generator_train_phase=generator_train_phase,
            discriminator_train_phase=discriminator_train_phase,
            generator_model_key=generator_model_key,
            discriminator_model_key=discriminator_model_key
        )
