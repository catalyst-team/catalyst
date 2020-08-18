from catalyst import utils
from catalyst.data.cv import BlurMixin, FlareMixin, RotateMixin

jpg_rgb_uri = (
    "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
    "/test_images/catalyst_icon.jpg"
)

image = utils.imread(jpg_rgb_uri)


def test_blur_mixin():
    """@TODO: Docs. Contribution is welcome."""
    global image
    image_dump = image.copy()

    mixin = BlurMixin()
    input = {"image": image_dump}  # noqa: WPS125
    output = mixin(input)

    assert mixin.input_key in output
    assert mixin.output_key in output
    assert output[mixin.input_key].shape == image_dump.shape

    assert 0 <= output[mixin.output_key] < mixin.blur_max


def test_flare_mixin():
    """@TODO: Docs. Contribution is welcome."""
    global image
    image_dump = image.copy()

    mixin = FlareMixin()
    input = {"image": image_dump}  # noqa: WPS125
    output = mixin(input)

    assert mixin.input_key in output
    assert mixin.output_key in output
    assert output[mixin.input_key].shape == image_dump.shape

    assert 0 <= output[mixin.output_key]


def test_rotate_mixin():
    """@TODO: Docs. Contribution is welcome."""
    global image
    image_dump = image.copy()

    mixin = RotateMixin()
    input = {"image": image_dump}  # noqa: WPS125
    output = mixin(input)

    assert mixin.input_key in output
    assert mixin.output_key in output
    assert output[mixin.input_key].shape == image_dump.shape

    assert 0 <= output[mixin.output_key] < 8
