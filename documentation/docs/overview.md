# Overview
---

**GlyContact** is an open-source Python package designed specifically for glycan structure analysis that can be entirely operated with glycan sequences (in any chosen nomenclature). 
It enables retrieving and processing three-dimensional glycan structures, performing sophisticated conformational analyses, and investigation of structure-function relationships. 


Many functionalities within **GlyContact** allow users to produce and retrieve heatmap plots or overlays of glycan sequences. Below are some code examples and their associated outputs.

<br>

## **Glycan Contact Maps**
---
Contact maps reveal the **spatial relationships** between monosaccharides in a glycan structure. These maps help identify which parts of the glycan are in close proximity, providing insights into potential functional regions.

```python
from glycontact.process import get_contact_tables
# Get monosaccharide contact tables
glycan = "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"
contact_tables = get_contact_tables(glycan, level="monosaccharide")

from glycontact.visualize import draw_contact_map
# Visualize the first contact map
draw_contact_map(contact_tables[0], size=1.0)
```

![png](img/README_4_0.png)

<br>

## **Surface Accessibility and Flexibility**
---
The solvent-accessible surface area (**SASA**) and **flexibility** of monosaccharides are crucial determinants of glycan-protein interactions. GlyContact calculates these properties and allows visualization of their distribution across the glycan structure.

```python
from glycontact.visualize import plot_glycan_score
plot_glycan_score(glycan, attribute="SASA")
```

![png](img/README_6_0.svg)

<br>

## **Glycosidic Torsion Angles**
---
Glycosidic **torsion angles (phi/psi)** determine the overall shape of glycans. GlyContact can analyze these angles across multiple structures to identify preferred conformations, **similar to protein Ramachandran plots**.

```python
from glycontact.visualize import ramachandran_plot
ramachandran_plot("GlcNAc(b1-4)GlcNAc")
```

![png](img/README_8_0.png)