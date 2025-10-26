# Stylization of Photographs using Tractography

<div align="center">
  <img src="test-images/cubain.jpg" alt="Original Image" width="600"/>
  <img src="test-images/out_cubain.png" alt="Stylized Image" width="600"/>
</div>

<br/>
Using the concepts from diffusion tensor imaging (DTI) and image processing,
this project aims to stylize photographs in painterly style.

For details, read the preprint [here](place/holder).

## Usage

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv run main.py INPUT_IMAGE OUTPUT_IMAGE [--params PARAMS_JSON] [--orientation-vector {structural,gradient}]
```

The optional `PARAMS_JSON` file can be used to specify parameters for the
algorithm. It should be a JSON file with the following structure:

```json
[
    { // First layer
         "sigma": float,               // Standard deviation for Gaussian smoothing
         "length_lines": float,        // Maximum length of lines in this layer
         "width": float,               // Width of lines in this layer
         "min_length": float,          // Minimum length of lines to draw
         "color_threshold": int        // Minimum color difference to draw a line
    },
    { // Second layer
        ...
    },
    ...
]
```

The `orientation-vector` argument specifies how to compute the orientation. If
`structural` it uses the primary eigenvector of the structure tensor (proposed improvement), if
`gradient` it uses the image gradient (for reference).

The settings used in the preprint can be found in `test-images/params.json`.