"""Transfer-learning model builder shared by all three architectures and both tasks.

Per spec: ImageNet-pretrained backbones, initially frozen, with progressive
layer-by-layer unfreezing for fine-tuning (unfreeze_top_layers). The same builder
serves the primary seven-class task (softmax head) and the secondary binary
malignant/benign task (sigmoid head) by varying `num_classes`.
"""

import tensorflow as tf

from src import config

_ARCHITECTURES = {
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
    "efficientnetb4": (
        tf.keras.applications.EfficientNetB4,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "vgg16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
}


def build_model(architecture: str, num_classes: int, task_name: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a frozen-backbone transfer-learning model.

    Returns (model, base) -- `base` is kept separate so unfreeze_top_layers can target
    it directly, rather than re-discovering it by layer name inside `model`.
    """
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}', expected one of {list(_ARCHITECTURES)}")

    base_cls, preprocess_fn = _ARCHITECTURES[architecture]
    input_shape = config.IMAGE_SIZE[architecture] + (3,)

    base = base_cls(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_fn(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{architecture}_{task_name}")
    return model, base


def preprocess_for(architecture: str, images: tf.Tensor) -> tf.Tensor:
    """Apply the architecture's own ImageNet preprocessing to raw-pixel-scale images.

    Exposed separately so XAI wrappers (src/xai/) can feed correctly-preprocessed
    images directly to a model's backbone without duplicating this mapping.
    """
    _, preprocess_fn = _ARCHITECTURES[architecture]
    return preprocess_fn(images)


def unfreeze_top_layers(base: tf.keras.Model, num_layers: int) -> None:
    """Unfreeze the top `num_layers` layers of the backbone for fine-tuning.

    BatchNormalization layers stay frozen even when unfrozen by position, since
    updating their running statistics on a small fine-tuning set is a common source of
    instability in transfer learning.
    """
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    for layer in base.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
