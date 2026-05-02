# Replication Code for "Visions of Politics: Mapping Ideological Resonance in QAnon’s Global Diffusion"

Code to reproduce my MA thesis on measuring ideological resonance through the alignment of field representations in QAnon discussions (USA/DE).

## Project Abstract

QAnon's global diffusion presents a compelling puzzle for understanding how ideologies spread across cultural boundaries. This study develops and tests a framework for measuring ideological resonance through the alignment of social ontologies: the meaningful representations of political fields that ideologies provide to their adherents.

I reconstruct latent political fields embedded in conspiracy discourses through named entity recognition, semantic motif extraction, PCA-based field representations, and word2vec semantic-axis analyses. The replication materials are organized into two analysis tracks: the original PCA analysis and the word2vec analysis used to compare semantic axes across English and German QAnon discourse.

## Repository Structure

- `data/`: Filtered motif CSVs (`motifs_en_filtered.csv`, `motifs_de_filtered.csv`, `motifs_media_filtered.csv`) and NER data (`qanon_ner.csv`, `media_ner.csv`). Large upstream Telegram/text-processing data are not included but can be provided upon request.
- `pca/`: Original PCA replication workflow, including `analysis.ipynb`, PCA utilities in `modules/`, entity mappings, translation mappings, and supplemental NER checks.
- `pca/modules/entities/`: Entity category, short-label, and `entities_mapping_2.json` files shared by the PCA and word2vec workflows.
- `w2v/preprocessing/`: Text-cleaning scripts and SLURM job file to process data before training the word2vec models.
- `w2v/training/`: CPU word2vec training scripts and SLURM job files for training the English and German models.
- `w2v/analysis/`: Word2vec notebooks, semantic-axis JSON definitions, plotting job file and plots, logs, and analysis helper modules.

## Requirements

The recommended setup is the conda environment in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ideological_resonance_thesis
```

A pip-oriented `requirements.txt` is also provided for reference, but the conda environment is preferred because the word2vec stack depends on compiled scientific Python packages.

## Replication Workflow

1. Run the PCA notebook from the repository root or from `pca/`:

```bash
jupyter notebook pca/analysis.ipynb
```

2. Prepare word2vec training data from the upstream text-processing exports:

```bash
sbatch w2v/preprocessing/preprocessing_final.sbatch
```

The preprocessing scripts expect the upstream raw CSVs as defined in the code. Replication data is not included in this repository but can be provided upon request. The generated cleaned CSVs are written to `w2v/data/` (not included) and are not versioned.

3. Train the English and German word2vec models:

```bash
sbatch w2v/training/train_w2v_cpu.sbatch
sbatch w2v/training/train_w2v_cpu_de.sbatch
```

The generated models are written to `w2v/models/2_w2v_min10/` and `w2v/models/2_w2v_min10_de/`. Model binaries/vectors are intentionally excluded because of their size.

4. Generate the word2vec plots:

```bash
sbatch w2v/analysis/w2v_plots.sbatch
```

The batch plotting script writes figures to `w2v/analysis/plots/` and uses the copied axis definitions in `w2v/analysis/axes_en.json` and `w2v/analysis/axes_de.json`. The `.out`/`.err` logs document the batch runs for transparency.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@mastersthesis{loertscher2026visions,
  author = {Loertscher, Pierre},
  title = {Visions of Politics: Mapping Ideological Resonance in QAnon's Global Diffusion},
  school = {University of Chicago},
  year = {2026}
}
```

Or cite the repository directly:

```bibtex
@software{loertscher2026qanon,
  author = {Loertscher, Pierre},
  title = {QAnon Diffusion Analysis Code and Replication Materials},
  url = {https://github.com/GitBagguetti/ideological_resonance_thesis},
  year = {2026}
}
```
