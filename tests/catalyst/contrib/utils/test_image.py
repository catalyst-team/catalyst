import pytest

from catalyst.settings import SETTINGS

if SETTINGS.cv_required:
    from catalyst.contrib.utils.image import imread


@pytest.mark.skipif(
    not (SETTINGS.cv_required), reason="No catalyst[cv] required",
)
def test_imread():
    """Tests ``imread`` functionality."""
    jpg_rgb_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon.jpg"
    )
    jpg_grs_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon_grayscale.jpg"
    )
    png_rgb_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon.png"
    )
    png_grs_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon_grayscale.png"
    )

    for uri in [jpg_rgb_uri, jpg_grs_uri, png_rgb_uri, png_grs_uri]:
        img = imread(uri)
        assert img.shape == (400, 400, 3)
        img = imread(uri, grayscale=True)
        assert img.shape == (400, 400, 1)


# def test_tensor_to_ndimage():
#     """Tests ``tensor_to_ndimage`` functionality."""
#     orig_images = np.random.randint(0, 255, (2, 20, 10, 3), np.uint8)
#
#     torch_images = torch.stack(
#         [normalize(to_tensor(im), _IMAGENET_MEAN, _IMAGENET_STD) for im in orig_images], dim=0,
#     )
#
#     byte_images = tensor_to_ndimage(torch_images, dtype=np.uint8)
#     float_images = tensor_to_ndimage(torch_images, dtype=np.float32)
#
#     assert np.allclose(byte_images, orig_images)
#     assert np.allclose(float_images, orig_images / 255, atol=1e-3, rtol=1e-3)
#
#     assert np.allclose(
#         tensor_to_ndimage(torch_images[0]), orig_images[0] / 255, atol=1e-3, rtol=1e-3,
#     )
