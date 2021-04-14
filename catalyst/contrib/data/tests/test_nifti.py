import os

import pytest

from catalyst.contrib.data.nifti.reader import NiftiReader
from catalyst.settings import SETTINGS

if SETTINGS.nifti_required:
    from nibabel.testing import data_path

    example_filename = os.path.join(data_path, "example4d.nii.gz")


@pytest.mark.skipif(not SETTINGS.nifti_required, reason="Niftly is not available")
@pytest.mark.parametrize("input_key, output_key", [("images", None), ("images", "outputs")])
def test_nifti_reader(input_key, output_key):
    """Minimal test for getting images from a nifti file"""
    test_annotations_dict = {input_key: example_filename}
    if output_key:
        nifti_reader = NiftiReader(input_key, output_key)
    else:
        nifti_reader = NiftiReader(input_key)
    output = nifti_reader(test_annotations_dict)
    assert isinstance(output, dict)
    if output_key is None:
        output_key = input_key
    assert output[output_key].shape == (128, 96, 24, 2)
