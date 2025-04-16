# 🌿 V-System: Vascular Lindenmayer Systems for Synthetic Vessel Generation

**V-System** is a Python-based framework for generating **synthetic 3D vascular networks** using **Lindenmayer Systems (L-Systems)** — a type of formal grammar traditionally used in plant modelling.

This implementation builds upon and extends the work by  
📖 [Galarreta-Valverde et al. (SPIE, 2013)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8669/86691I/Three-dimensional-synthetic-blood-vessel-generation-using-stochastic-L-systems/10.1117/12.2007532.full?SSO=1),  
who introduced stochastic rules and biological realism into traditional L-Systems for vessel modelling.

---

## 🧠 System Overview

V-System simulates vessel formation through three stages:

1. **Grammar Generation**: Build a rule-based string using iterative stochastic L-System rules.
2. **Geometry Construction**: Convert the string into 3D coordinates for branching and direction.
3. **Voxelization**: Perform voxel traversal to create a 3D binary mask (volumetric image).

![Example](https://github.com/psweens/V-System/blob/master/Lnet_Generations.jpg)

---

## 📦 Installation

V-System runs on Python 3.9 and has been tested on:
- Ubuntu 18.04 LTS
- macOS Big Sur  
It should also work on most Linux and Windows systems.

Clone the repository:

```bash
git clone https://github.com/psweens/V-System.git
```

Install required packages:

```bash
pip install scikit-image opencv-python bezier matplotlib itk
```

Or use the full requirements file:

📄 [`REQUIREMENTS.txt`](https://github.com/psweens/V-System/blob/master/REQUIREMENTS.txt)

---

## 🔬 Applications

- Generating realistic 3D vascular structures for simulation.
- Training datasets for segmentation algorithms (e.g. VAN-GAN).
- Creating synthetic benchmarks for vascular imaging pipelines.
- Testing robustness of deep learning models under anatomical variability.

---

## 📖 Citation

Please cite the following if you use V-System in your research:

**Version 1.0**  
> [Quantification of vascular networks in photoacoustic mesoscopy](https://www.sciencedirect.com/science/article/pii/S221359792200026X)  
> Emma L. Brown, Thierry L. Lefebvre, Paul W. Sweeney et al.

**Version 2.0**  
> [Unsupervised Segmentation of 3D Microvascular Photoacoustic Images Using Deep Generative Learning](https://doi.org/10.1002/advs.202402195)  
> Paul W. Sweeney et al., *Advanced Science*, 2024

---

## 🙏 Acknowledgements

This project builds upon foundational work by  
🎓 Miguel A. Galarreta-Valverde – [PhD Thesis (USP)](https://teses.usp.br/teses/disponiveis/45/45134/tde-30112012-172822/pt-br.php)

---

## 📬 Contact

For bugs, ideas, or contributions — feel free to open an issue or reach out through the [GitHub repository](https://github.com/psweens/V-System).

---
