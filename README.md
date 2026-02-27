# Replication Code for "Visions of Politics: Mapping Ideological Resonance in QAnon’s Global Diffusion"
Code to reproduce my MA thesis on measuring ideological resonance through the alignment of field representations in QAnon discussions (USA/GE).

## Project Abstract
QAnon's global diffusion presents a compelling puzzle for understanding how ideologies spread across cultural boundaries. This study develops and tests a framework for measuring ideological resonance through the alignment of social ontologies – the meaningful representations of political fields that ideologies provide to their adherents. 

I argue that conspiracy theories, like other ideologies, serve as symbolic reconstructions of political space that help believers navigate complex alliance structures and identify relevant friends and foes. Using socio-symbolic network analysis on 28.4 million Telegram posts from U.S.-American and German QAnon communities (2019-2022), I reconstruct the latent political fields embedded in conspiracy discourses through named entity recognition and semantic motif extraction. Two complementary analytical approaches – standard and subjectively 'distorted' principal component analysis – reveal how conspiracy theorists organize political actors through symbolic action associations. 

Below, two plots show a comparison of the political fields represented in American versus German QAnon discourses, with actor distributions distorted by unique actor-action pairings emphasized in QAnon discourses:

<p align="center">
  <img src="plots/Frequency-distorted PCA EN logscaled.png" alt="Frequency-distorted PCA EN (log-scaled)" width="45%" />
  <img src="plots/Frequency-distorted PCA DE logscaled.png" alt="Frequency-distorted PCA DE (log-scaled)" width="45%" />
</p>

The results demonstrate strong alignment between American and German QAnon affective worldviews across three dimensions: comparable actor groups (local elites, alternative politicians, spiritual figures), similar organizing axes (political-epistemic conflict, institutional-individual distinctions), and homologous distributional patterns within reconstructed political fields. 

However, my analyses also revealed important local adaptations, particularly the diminished role of religious figures in German discourses and the incorporation of European institutions into the conspiracy's antagonist coalition. 

These findings suggest that resonance involves selective compatibility rather than wholesale ideological transfer, with successful diffusion requiring structural similarities that allow for meaningful local translation. These findings contribute to diffusion theory by providing a measurable framework for understanding how cultural templates achieve cross-contextual appeal.

## Repo Structure

- **`ideological_resonance_thesis/`** : Root repository for reproducing the thesis analysis on ideological resonance in QAnon discourses (USA/DE).
- **`data/`** : Input data: filtered motif CSVs (`motifs_en_filtered.csv`, `motifs_de_filtered.csv`, `motifs_media_filtered.csv`) and NER data (`qanon_ner.csv`, `media_ner.csv`). Due to large file sizes, the data is available on request only.
- **`modules/`** : Python utilities for PCA, actor-action matrices, and visualizations; contains `analysis_util.py`.
- **`modules/translations/`** : JSON mappings for actions and verb translations (`actions_mapping.json`, `translations_de.json`, `translations_it.json`); Excel translation outputs are written here.
- **`plots/`** : Figures used for README demonstration (frequency-distorted PCA plots).
- **`analysis.ipynb`** : Main Jupyter notebook for running the analysis and producing the figures.

## Requirements

Use either **`requirements.txt`** (pip) or **`environment.yml`** (conda) to recreate the environment. Both target Python 3.14.3.

- **pip:** `pip install -r requirements.txt`
- **conda:** `conda env create -f environment.yml`

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
