![Alt text](img/TOC.png)

# Hypershape recognition (HSR): a general framework for moment-based similarity measures

 HSR is a versatile, moment-based similarity measure tailored for three-dimensional (3D) chemical representations annotated with atomic features. It enhances the robustness and versatility of the Ultrafast Shape Recognition (USR) method by incorporating multidimensional features for each atom, such as protons, neutrons, and formal charges.

## Getting Started

### Installing HSR

You can install HSR using either pip or conda:

```bash
pip install hsr
```
or 

```bash
conda install hsr -c conda-forge
```

### Build from source

Clone this repository on your machine. Move inside it and create the conda environment:

```bash
conda env create -f environment.yml
conda activate HSR_devel
```
Verify the correct creation of the environment by running:

```bash
pytest
```
To use HSR from CLI you can run:
```bash
python -m hsr.hsr_cli 
```
If HSR is installed with pip or conda, the above command is replaced by the simple use of ``hsr``

### Basic Usage

Run the folowing command to get help in using HSR from CLI:

```bash
hsr -h
```

For a detailed overview of HSR's methodology check our [documentation](https://denoptim-project.github.io/HSR/).


### Licensing

HSR is licensed under the [GNU Affero General Public License Version 3](https://www.gnu.org/licenses/agpl-3.0.html). For more details, see the LICENSE file.

### Citing HSR

If you use HSR in your research, please cite it as follows:

[TODO: Add citation. Publishing in progress!]

### Contributing

Contributions to HSR are welcome! Please read our [Contributing Guidelines](https://denoptim-project.github.io/HSR/CONTRIBUTING.html) for information on how to get started.

